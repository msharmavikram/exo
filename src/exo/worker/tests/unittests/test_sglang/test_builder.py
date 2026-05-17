import time
from collections.abc import Mapping
from dataclasses import dataclass

from pytest import MonkeyPatch

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import BoundInstance, InstanceId, SglangInstance
from exo.shared.types.worker.runner_response import CancelledResponse, FinishedResponse
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import TensorShardMetadata
from exo.utils.channels import mp_channel
from exo.worker.engines.sglang.builder import SglangBuilder, SglangEngine


@dataclass
class CapturedPost:
    path: str
    json: dict[str, object]


@dataclass
class FakeProcess:
    cmd: list[str]
    returncode: int | None = None
    terminated: bool = False
    killed: bool = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode or 0


class FakeResponse:
    status_code: int

    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeClient:
    def __init__(
        self, *, base_url: str = "http://127.0.0.1:30000", timeout: float | None = None
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.posts: list[CapturedPost] = []
        self.gets = 0
        self.closed = False

    def get(self, path: str, *, timeout: float | None = None) -> FakeResponse:
        self.gets += 1
        return FakeResponse({})

    def post(self, path: str, *, json: Mapping[str, object]) -> FakeResponse:
        self.posts.append(CapturedPost(path=path, json=dict(json)))
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {"content": "hello from sglang"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                },
            }
        )

    def close(self) -> None:
        self.closed = True


def _model_card(
    *,
    model_id: str = "nvidia/Llama-3.1-8B-Instruct-FP4",
    quantization: str = "",
) -> ModelCard:
    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_bytes(100),
        n_layers=2,
        hidden_size=16,
        supports_tensor=True,
        quantization=quantization,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.SglangCuda],
        trust_remote_code=True,
    )


def _bound_instance(
    *,
    node_id: NodeId | None = None,
    rank: int = 0,
    world_size: int = 1,
    dist_addr: str = "0.0.0.0:25000",
    model_card: ModelCard | None = None,
) -> BoundInstance:
    node_id = node_id or NodeId("node-0")
    runner_id = RunnerId(f"runner-{rank}")
    card = model_card or _model_card()
    shard = TensorShardMetadata(
        model_card=card,
        device_rank=rank,
        world_size=world_size,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )
    instance = SglangInstance(
        instance_id=InstanceId("instance-0"),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            runner_to_shard={runner_id: shard},
            node_to_runner={node_id: runner_id},
        ),
        service_port=30000,
        dist_init_port=25000,
        dist_init_addrs={node_id: dist_addr},
    )
    return BoundInstance(
        instance=instance,
        bound_runner_id=runner_id,
        bound_node_id=node_id,
    )


def _task(task_id: TaskId | None = None) -> TextGeneration:
    return TextGeneration(
        task_id=task_id or TaskId("task-0"),
        instance_id=InstanceId("instance-0"),
        command_id=CommandId("command-0"),
        task_params=TextGenerationTaskParams(
            model=ModelId("nvidia/Llama-3.1-8B-Instruct-FP4"),
            instructions=InputMessageContent("be concise"),
            input=[
                InputMessage(
                    role="user",
                    content=InputMessageContent("say hello"),
                )
            ],
            max_output_tokens=12,
            temperature=0.2,
            top_p=0.9,
        ),
    )


def test_sglang_builder_launches_single_dgx_spark_command(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_SGLANG_SKIP_HEALTH_WAIT", "1")
    monkeypatch.setenv("EXO_SGLANG_LAUNCH_CMD", "python -m sglang.launch_server")
    monkeypatch.delenv("EXO_SGLANG_EXTRA_ARGS", raising=False)
    processes: list[FakeProcess] = []

    def process_factory(args: list[str], *, env: Mapping[str, str]) -> FakeProcess:
        assert env
        process = FakeProcess(cmd=args)
        processes.append(process)
        return process

    _, cancel_receiver = mp_channel[TaskId]()
    builder = SglangBuilder(
        model_id=ModelId("nvidia/Llama-3.1-8B-Instruct-FP4"),
        cancel_receiver=cancel_receiver,
        process_factory=process_factory,
    )

    list(builder.load(_bound_instance(model_card=_model_card(quantization="NVFP4"))))

    cmd = processes[0].cmd
    assert cmd[:3] == ["python", "-m", "sglang.launch_server"]
    assert "--model-path" in cmd
    assert cmd[cmd.index("--port") + 1] == "30000"
    assert cmd[cmd.index("--tp") + 1] == "1"
    assert cmd[cmd.index("--attention-backend") + 1] == "flashinfer"
    assert cmd[cmd.index("--mem-fraction-static") + 1] == "0.75"
    assert "--trust-remote-code" in cmd
    assert cmd[cmd.index("--quantization") + 1] == "modelopt_fp4"


def test_sglang_builder_defaults_gpt_oss_attention_to_triton(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_SGLANG_SKIP_HEALTH_WAIT", "1")
    monkeypatch.setenv("EXO_SGLANG_LAUNCH_CMD", "python -m sglang.launch_server")
    monkeypatch.delenv("EXO_SGLANG_ATTENTION_BACKEND", raising=False)
    monkeypatch.delenv("EXO_SGLANG_EXTRA_ARGS", raising=False)
    processes: list[FakeProcess] = []

    def process_factory(args: list[str], *, env: Mapping[str, str]) -> FakeProcess:
        assert env
        process = FakeProcess(cmd=args)
        processes.append(process)
        return process

    _, cancel_receiver = mp_channel[TaskId]()
    builder = SglangBuilder(
        model_id=ModelId("openai/gpt-oss-20b"),
        cancel_receiver=cancel_receiver,
        process_factory=process_factory,
    )

    list(
        builder.load(
            _bound_instance(model_card=_model_card(model_id="openai/gpt-oss-20b"))
        )
    )

    cmd = processes[0].cmd
    assert cmd[cmd.index("--attention-backend") + 1] == "triton"


def test_sglang_builder_adds_multi_node_dist_args(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("EXO_SGLANG_LAUNCH_CMD", "python -m sglang.launch_server")
    monkeypatch.delenv("EXO_SGLANG_EXTRA_ARGS", raising=False)
    processes: list[FakeProcess] = []

    def process_factory(args: list[str], *, env: Mapping[str, str]) -> FakeProcess:
        assert env
        process = FakeProcess(cmd=args)
        processes.append(process)
        return process

    _, cancel_receiver = mp_channel[TaskId]()
    node_id = NodeId("node-1")
    builder = SglangBuilder(
        model_id=ModelId("nvidia/Llama-3.1-8B-Instruct-FP4"),
        cancel_receiver=cancel_receiver,
        process_factory=process_factory,
    )

    list(
        builder.load(
            _bound_instance(
                node_id=node_id,
                rank=1,
                world_size=2,
                dist_addr="169.254.0.1:25000",
            )
        )
    )

    cmd = processes[0].cmd
    assert cmd[cmd.index("--tp") + 1] == "2"
    assert cmd[cmd.index("--dist-init-addr") + 1] == "169.254.0.1:25000"
    assert cmd[cmd.index("--nnodes") + 1] == "2"
    assert cmd[cmd.index("--node-rank") + 1] == "1"


def test_sglang_engine_maps_chat_completion_to_token_chunk() -> None:
    _, cancel_receiver = mp_channel[TaskId]()
    client = FakeClient()

    def client_factory(*, base_url: str, timeout: float | None) -> FakeClient:
        return client

    engine = SglangEngine(
        model_id=ModelId("nvidia/Llama-3.1-8B-Instruct-FP4"),
        base_url="http://127.0.0.1:30000",
        device_rank=0,
        cancel_receiver=cancel_receiver,
        client_factory=client_factory,
    )

    task = _task()
    engine.submit(task)
    results = list(engine.step())

    token = results[0][1]
    assert isinstance(token, TokenChunk)
    assert token.text == "hello from sglang"
    assert token.finish_reason == "stop"
    assert token.stats is not None
    assert token.stats.prompt_tokens == 7
    assert token.stats.generation_tokens == 3
    assert token.stats.prompt_tps > 0
    assert token.stats.generation_tps > 0
    assert isinstance(results[1][1], FinishedResponse)
    payload = client.posts[0].json
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert messages[0] == {"role": "system", "content": "be concise"}
    assert messages[1] == {"role": "user", "content": "say hello"}
    assert payload["max_tokens"] == 12
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.9


def test_sglang_engine_nonzero_rank_finishes_without_http() -> None:
    _, cancel_receiver = mp_channel[TaskId]()

    def fail_client_factory(*, base_url: str, timeout: float | None) -> FakeClient:
        raise AssertionError("no http")

    engine = SglangEngine(
        model_id=ModelId("nvidia/Llama-3.1-8B-Instruct-FP4"),
        base_url="http://127.0.0.1:30000",
        device_rank=1,
        cancel_receiver=cancel_receiver,
        client_factory=fail_client_factory,
    )

    engine.submit(_task())
    results = list(engine.step())

    assert len(results) == 1
    assert isinstance(results[0][1], FinishedResponse)


def test_sglang_engine_honors_cancelled_task() -> None:
    cancel_sender, cancel_receiver = mp_channel[TaskId]()
    task_id = TaskId("task-cancelled")
    cancel_sender.send(task_id)
    time.sleep(0.05)

    def fail_client_factory(*, base_url: str, timeout: float | None) -> FakeClient:
        raise AssertionError("no http")

    engine = SglangEngine(
        model_id=ModelId("nvidia/Llama-3.1-8B-Instruct-FP4"),
        base_url="http://127.0.0.1:30000",
        device_rank=0,
        cancel_receiver=cancel_receiver,
        client_factory=fail_client_factory,
    )

    engine.submit(_task(task_id))
    results = list(engine.step())

    assert len(results) == 1
    assert isinstance(results[0][1], CancelledResponse)
