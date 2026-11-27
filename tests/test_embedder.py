from types import SimpleNamespace

import httpx2
import pytest
from mistralai.client.errors import NoResponseError, SDKError

from trip.llm import MistralEmbedder, _is_transient
from trip.models import EmbeddingUnavailable


def status_error(status: int) -> SDKError:
    return SDKError("API error occurred", httpx2.Response(status))


class FakeEmbeddingsAPI:
    """Stands for client.embeddings: raises the queued errors, then answers.

    The vector of text "7" is [7.0]. Items come back in reverse order, each with its index.
    """

    def __init__(self, *errors: Exception, drop: int = 0) -> None:
        self._errors = list(errors)
        self._drop = drop
        self.batches: list[list[str]] = []

    async def create_async(self, *, model: str, inputs: list[str]) -> SimpleNamespace:
        self.batches.append(inputs)
        if self._errors:
            raise self._errors.pop(0)
        data = [SimpleNamespace(index=i, embedding=[float(t)]) for i, t in enumerate(inputs)]
        return SimpleNamespace(
            data=data[::-1][self._drop :], usage=SimpleNamespace(total_tokens=len(inputs))
        )


def make_embedder(api: FakeEmbeddingsAPI) -> MistralEmbedder:
    client = SimpleNamespace(embeddings=api)
    return MistralEmbedder(client=client, model="mistral-embed", batch_size=32)  # type: ignore[arg-type]


async def test_texts_are_embedded_in_batches_and_keep_their_order():
    api = FakeEmbeddingsAPI()
    texts = [str(i) for i in range(70)]

    vectors = await make_embedder(api).embed(texts)

    assert [len(b) for b in api.batches] == [32, 32, 6]
    assert vectors == [[float(i)] for i in range(70)]


async def test_no_text_makes_no_call():
    api = FakeEmbeddingsAPI()

    assert await make_embedder(api).embed([]) == []
    assert api.batches == []


async def test_a_rate_limited_batch_is_retried():
    api = FakeEmbeddingsAPI(status_error(429))

    assert await make_embedder(api).embed(["1"]) == [[1.0]]
    assert len(api.batches) == 2


@pytest.mark.parametrize("error", [status_error(401), NoResponseError()])
async def test_a_definitive_error_becomes_embedding_unavailable(error: Exception):
    api = FakeEmbeddingsAPI(error)

    with pytest.raises(EmbeddingUnavailable):
        await make_embedder(api).embed(["1"])

    assert len(api.batches) == 1


async def test_a_missing_vector_is_an_error():
    with pytest.raises(EmbeddingUnavailable):
        await make_embedder(FakeEmbeddingsAPI(drop=1)).embed(["1", "2"])


@pytest.mark.parametrize(
    "error",
    [
        status_error(429),
        status_error(500),
        status_error(503),
        httpx2.ReadTimeout("timed out"),
        httpx2.ConnectError("connection refused"),
    ],
)
def test_rate_limit_server_and_transport_errors_are_transient(error: Exception):
    assert _is_transient(error)


@pytest.mark.parametrize("status", [400, 401, 422])
def test_client_errors_are_definitive(status: int):
    assert not _is_transient(status_error(status))
