import contextlib
import os
import shlex
import subprocess
import sys
import time
from collections import deque
from collections.abc import Generator, Iterable, Mapping
from dataclasses import dataclass, field
from typing import BinaryIO, Literal, Protocol, cast

import httpx

from exo.api.types import ToolCallItem
from exo.download.download_utils import build_model_path
from exo.shared.models.model_cards import ModelCard, ModelId
from exo.shared.types.chunks import Chunk, ErrorChunk, TokenChunk, ToolCallChunk
from exo.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    GenerationTask,
    TaskId,
    TextGeneration,
)
from exo.shared.types.text_generation import (
    ChatTemplateValue,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import BoundInstance, SglangInstance
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
    ModelLoadingResponse,
)
from exo.utils.channels import MpReceiver
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.base import Builder, Engine
from exo.worker.runner.bootstrap import logger

type TokenFinishReason = Literal["stop", "length", "content_filter"]


class SglangProcess(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class SglangResponse(Protocol):
    @property
    def status_code(self) -> int: ...

    def json(self) -> object: ...

    def raise_for_status(self) -> None: ...


class SglangHttpClient(Protocol):
    def get(self, path: str, *, timeout: float | None = None) -> SglangResponse: ...

    def post(self, path: str, *, json: Mapping[str, object]) -> SglangResponse: ...

    def close(self) -> None: ...


class ProcessFactory(Protocol):
    def __call__(self, args: list[str], *, env: Mapping[str, str]) -> SglangProcess: ...


class ClientFactory(Protocol):
    def __call__(self, *, base_url: str, timeout: float | None) -> SglangHttpClient: ...


@dataclass
class HttpxSglangResponse:
    response: httpx.Response

    @property
    def status_code(self) -> int:
        return self.response.status_code

    def json(self) -> object:
        return cast(object, self.response.json())

    def raise_for_status(self) -> None:
        self.response.raise_for_status()


@dataclass
class HttpxSglangClient:
    client: httpx.Client

    def get(self, path: str, *, timeout: float | None = None) -> SglangResponse:
        return HttpxSglangResponse(self.client.get(path, timeout=timeout))

    def post(self, path: str, *, json: Mapping[str, object]) -> SglangResponse:
        return HttpxSglangResponse(self.client.post(path, json=dict(json)))

    def close(self) -> None:
        self.client.close()


def _default_process_factory(
    args: list[str], *, env: Mapping[str, str]
) -> SglangProcess:
    return subprocess.Popen(args, env=dict(env))


def _default_client_factory(
    *, base_url: str, timeout: float | None
) -> SglangHttpClient:
    return HttpxSglangClient(httpx.Client(base_url=base_url, timeout=timeout))


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _jsonable(value: ChatTemplateValue) -> object:
    if isinstance(value, dict):
        mapping = cast(dict[str, ChatTemplateValue], value)
        return {str(k): _jsonable(v) for k, v in mapping.items()}
    if isinstance(value, list):
        items = cast(list[ChatTemplateValue], value)
        return [_jsonable(v) for v in items]
    return value


def _messages_from_params(params: TextGenerationTaskParams) -> list[dict[str, object]]:
    if params.chat_template_messages is not None:
        return [
            {str(k): _jsonable(v) for k, v in message.items()}
            for message in params.chat_template_messages
        ]

    messages: list[dict[str, object]] = []
    if params.instructions:
        messages.append({"role": "system", "content": str(params.instructions)})

    for message in params.input:
        messages.append({"role": message.role, "content": str(message.content)})

    return messages


def _sampling_payload(params: TextGenerationTaskParams) -> dict[str, object]:
    payload: dict[str, object] = {}
    if params.max_output_tokens is not None:
        payload["max_tokens"] = params.max_output_tokens
    if params.temperature is not None:
        payload["temperature"] = params.temperature
    if params.top_p is not None:
        payload["top_p"] = params.top_p
    if params.stop is not None:
        payload["stop"] = params.stop
    if params.seed is not None:
        payload["seed"] = params.seed
    if params.tools is not None:
        payload["tools"] = params.tools
    if params.logprobs:
        payload["logprobs"] = True
    if params.top_logprobs is not None:
        payload["top_logprobs"] = params.top_logprobs
    if params.frequency_penalty is not None:
        payload["frequency_penalty"] = params.frequency_penalty
    if params.presence_penalty is not None:
        payload["presence_penalty"] = params.presence_penalty
    return payload


def _finish_reason(value: object) -> TokenFinishReason | None:
    if value in ("stop", "length", "content_filter"):
        return value
    return None


def _object_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return None


def _tool_calls(message: Mapping[str, object]) -> list[ToolCallItem]:
    parsed: list[ToolCallItem] = []
    raw_tool_calls = message.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return parsed

    for raw_item in cast(list[object], raw_tool_calls):
        item = _object_mapping(raw_item)
        if item is None:
            continue
        function = _object_mapping(item.get("function")) or {}
        parsed.append(
            ToolCallItem(
                id=str(item.get("id", "")),
                name=str(function.get("name", "")),
                arguments=str(function.get("arguments", "")),
            )
        )
    return parsed


def _sglang_quantization_arg(model_card: ModelCard) -> str | None:
    override = os.environ.get("EXO_SGLANG_QUANTIZATION")
    if override:
        return override

    quantization = model_card.quantization.lower()
    model_id = model_card.model_id.lower()
    if "nvfp4" in quantization or "nvfp4" in model_id:
        return "modelopt_fp4"
    return None


@dataclass
class SglangBuilder(Builder):
    model_id: ModelId
    cancel_receiver: MpReceiver[TaskId]
    process_factory: ProcessFactory = _default_process_factory
    client_factory: ClientFactory = _default_client_factory
    process: SglangProcess | None = None
    base_url: str | None = None
    _device_rank: int = 0

    def connect(self, bound_instance: BoundInstance) -> None:
        if not isinstance(bound_instance.instance, SglangInstance):
            raise TypeError("SglangBuilder requires a SglangInstance")

    def load(self, bound_instance: BoundInstance) -> Generator[ModelLoadingResponse]:
        if not isinstance(bound_instance.instance, SglangInstance):
            raise TypeError("SglangBuilder requires a SglangInstance")
        if self.process is not None:
            yield ModelLoadingResponse(layers_loaded=1, total=1)
            return

        cmd = self._launch_command(bound_instance)
        env = os.environ.copy()
        logger.info(f"Launching SGLang: {shlex.join(cmd)}")
        self.process = self.process_factory(cmd, env=env)
        self.base_url = (
            f"http://{os.environ.get('EXO_SGLANG_CLIENT_HOST', '127.0.0.1')}:"
            f"{bound_instance.instance.service_port}"
        )
        yield ModelLoadingResponse(layers_loaded=1, total=1)

    def build(self) -> Engine:
        assert self.base_url is not None
        return SglangEngine(
            model_id=self.model_id,
            base_url=self.base_url,
            device_rank=self._device_rank,
            cancel_receiver=self.cancel_receiver,
            process=self.process,
            client_factory=self.client_factory,
        )

    def close(self) -> None:
        if self.process is None:
            return
        _terminate_process(self.process)
        self.process = None

    def _launch_command(self, bound_instance: BoundInstance) -> list[str]:
        instance = bound_instance.instance
        assert isinstance(instance, SglangInstance)

        self._device_rank = bound_instance.bound_shard.device_rank
        model_card = bound_instance.bound_shard.model_card
        model_path = build_model_path(model_card.model_id)
        model_arg = str(model_path) if model_path.exists() else str(model_card.model_id)

        command_prefix = shlex.split(
            os.environ.get(
                "EXO_SGLANG_LAUNCH_CMD",
                f"{sys.executable} -m sglang.launch_server",
            )
        )
        cmd = [
            *command_prefix,
            "--model-path",
            model_arg,
            "--host",
            os.environ.get("EXO_SGLANG_HOST", "0.0.0.0"),
            "--port",
            str(instance.service_port),
            "--tp",
            str(bound_instance.bound_shard.world_size),
            "--mem-fraction-static",
            os.environ.get("EXO_SGLANG_MEM_FRACTION_STATIC", "0.75"),
        ]

        attention_backend = os.environ.get("EXO_SGLANG_ATTENTION_BACKEND", "flashinfer")
        if attention_backend:
            cmd.extend(["--attention-backend", attention_backend])

        if model_card.trust_remote_code:
            cmd.append("--trust-remote-code")

        if quantization := _sglang_quantization_arg(model_card):
            cmd.extend(["--quantization", quantization])

        if bound_instance.bound_shard.world_size > 1:
            cmd.extend(
                [
                    "--dist-init-addr",
                    instance.dist_init_addrs[bound_instance.bound_node_id],
                    "--nnodes",
                    str(bound_instance.bound_shard.world_size),
                    "--node-rank",
                    str(bound_instance.bound_shard.device_rank),
                ]
            )

        if extra_args := os.environ.get("EXO_SGLANG_EXTRA_ARGS"):
            cmd.extend(shlex.split(extra_args))

        return cmd


@dataclass(eq=False)
class SglangEngine(Engine):
    model_id: ModelId
    base_url: str
    device_rank: int
    cancel_receiver: MpReceiver[TaskId]
    process: SglangProcess | None = None
    client_factory: ClientFactory = _default_client_factory

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _queue: deque[TextGeneration] = field(default_factory=deque, init=False)
    _client: SglangHttpClient | None = field(default=None, init=False)

    def warmup(self) -> None:
        if self.device_rank != 0:
            return
        if os.environ.get("EXO_SGLANG_SKIP_HEALTH_WAIT"):
            return

        timeout_seconds = _env_int("EXO_SGLANG_STARTUP_TIMEOUT", 600)
        deadline = time.monotonic() + timeout_seconds
        client = self._get_client()
        while time.monotonic() < deadline:
            self._raise_if_process_exited()
            try:
                response = client.get("/health", timeout=5)
                if response.status_code < 500:
                    logger.info("SGLang health check passed")
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1)

        raise TimeoutError(f"SGLang did not become healthy within {timeout_seconds}s")

    def submit(self, task: GenerationTask) -> None:
        if not isinstance(task, TextGeneration):
            raise ValueError("SGLang engine only supports text generation tasks")
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._queue.append(task)

    def step(
        self,
    ) -> Iterable[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]]:
        self._collect_cancellations()
        if not self._queue:
            return []

        task = self._queue.popleft()
        if self.should_cancel(task.task_id):
            return [(task.task_id, CancelledResponse())]

        if self.device_rank != 0:
            return [(task.task_id, FinishedResponse())]

        try:
            chunks = self._complete(task)
        except Exception as e:
            logger.opt(exception=e).warning("SGLang request failed")
            chunks = [
                ErrorChunk(
                    model=self.model_id,
                    error_message=str(e),
                )
            ]
        return [(task.task_id, chunk) for chunk in chunks] + [
            (task.task_id, FinishedResponse())
        ]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        if self.process is not None:
            _terminate_process(self.process)
            self.process = None

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        raise NotImplementedError("SGLang engine does not expose exo prefill transfer")

    def _complete(self, task: TextGeneration) -> list[Chunk]:
        if task.task_params.images:
            raise ValueError("SGLang vision requests are not wired through exo yet")

        self._raise_if_process_exited()
        payload = self._openai_chat_payload(task.task_params)
        response = self._get_client().post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = _object_mapping(response.json())
        if data is None:
            raise ValueError("SGLang returned a non-object response")

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("SGLang response did not include choices")

        choice_items = cast(list[object], choices)
        choice = _object_mapping(choice_items[0])
        if choice is None:
            raise ValueError("SGLang response choice was not an object")

        message = _object_mapping(choice.get("message")) or {}

        tools = _tool_calls(message)
        if tools:
            tool_chunk: Chunk = ToolCallChunk(
                model=self.model_id,
                tool_calls=tools,
                usage=None,
            )
            return [tool_chunk]

        content = message.get("content", choice.get("text", ""))
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in cast(list[object], content):
                part_mapping = _object_mapping(part)
                if part_mapping is not None:
                    text_parts.append(str(part_mapping.get("text", "")))
                else:
                    text_parts.append(str(part))
            text = "".join(text_parts)
        else:
            text = "" if content is None else str(content)

        token_chunk: Chunk = TokenChunk(
            model=self.model_id,
            text=text,
            token_id=-1,
            usage=None,
            finish_reason=_finish_reason(choice.get("finish_reason")),
        )
        return [token_chunk]

    def _openai_chat_payload(
        self, params: TextGenerationTaskParams
    ) -> dict[str, object]:
        return {
            "model": str(params.model),
            "messages": _messages_from_params(params),
            "stream": False,
            **_sampling_payload(params.with_card_sampling_defaults()),
        }

    def _get_client(self) -> SglangHttpClient:
        if self._client is None:
            self._client = self.client_factory(base_url=self.base_url, timeout=None)
        return self._client

    def _collect_cancellations(self) -> None:
        for task_id in self.cancel_receiver.collect():
            self._cancelled_tasks.add(task_id)

    def _raise_if_process_exited(self) -> None:
        if self.process is None:
            return
        exit_code = self.process.poll()
        if exit_code is not None:
            raise RuntimeError(f"SGLang process exited with code {exit_code}")


def _terminate_process(process: SglangProcess) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    with contextlib.suppress(TimeoutError, subprocess.TimeoutExpired):
        process.wait(timeout=_env_float("EXO_SGLANG_SHUTDOWN_TIMEOUT", 10.0))
        return
    process.kill()
    with contextlib.suppress(Exception):
        process.wait(timeout=5)
