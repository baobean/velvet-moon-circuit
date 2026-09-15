import pytest
from PIL import Image

from ragregen import draft


class FakePipe:
    """Mirrors FluxKontextPipeline.__call__.

    Two things this fake must get right, both learned the hard way:

    1. The real signature starts (image, prompt, ...) -- `image` first, not
       `prompt`. A fake accepting `prompt` positionally accepts a call the
       real pipeline rejects with "Provide either `prompt` or `prompt_embeds`",
       which is exactly how a green suite still crashed on the first case of a
       real run.
    2. It records a *snapshot* of the generator state at each call. Holding the
       generator object instead makes the determinism test vacuous: a Drafter
       that wrongly shares one generator hands back the same object twice, so
       comparing the objects' states after the fact always succeeds.
    """

    def __init__(self, consume=False):
        self.calls = []
        self.seeds = []
        self.states = []
        self.images = []
        self.guidances = []
        self._consume = consume

    def __call__(self, image=None, prompt=None, num_inference_steps=None,
                 generator=None, guidance_scale=None, **kw):
        import torch

        if prompt is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave "
                "both `prompt` and `prompt_embeds` undefined.")
        self.calls.append((prompt, num_inference_steps))
        self.images.append(image)
        self.guidances.append(guidance_scale)
        self.seeds.append(generator.initial_seed())
        self.states.append(generator.get_state().clone())
        if self._consume:
            # A real diffusion pipeline draws from the generator; an
            # incorrectly shared one would then be advanced for the next call.
            torch.randn(4, generator=generator)

        class Out:
            images = [Image.new("RGB", (64, 64), (5, 5, 5))]

        return Out()


def test_draft_returns_an_image_and_forwards_steps():
    pipe = FakePipe()
    d = draft.Drafter(pipe, steps=28, seed=0)
    img = d.draft("an Amur leopard")
    assert isinstance(img, Image.Image)
    assert pipe.calls[0] == ("an Amur leopard", 28)


def test_draft_forwards_prompt_and_steps_each_call():
    """Every call carries the configured prompt and step count.

    The plan asserted only `len(pipe.calls) == 2`, which passes even if the
    prompt or steps were dropped on the second call.
    """
    pipe = FakePipe()
    d = draft.Drafter(pipe, steps=4, seed=7)
    d.draft("a")
    d.draft("a")
    assert pipe.calls == [("a", 4), ("a", 4)]


def test_same_seed_reseeds_the_generator_on_every_call():
    """Two drafts at one seed must start from the same generator state.

    A Drafter that built its generator once in __init__ would advance it
    between calls, so the same case re-drafted in one process would not be
    reproducible -- and the gate's fail rate would not be re-checkable.
    """
    pipe = FakePipe(consume=True)
    d = draft.Drafter(pipe, steps=4, seed=7)
    d.draft("a")
    d.draft("a")
    first, second = pipe.states
    assert pipe.seeds == [7, 7]
    assert first.equal(second), (
        "the second draft started from an advanced generator state; the "
        "Drafter is sharing one generator across calls instead of reseeding")


def test_different_seeds_produce_different_generator_states():
    pipe = FakePipe()
    draft.Drafter(pipe, steps=4, seed=1).draft("a")
    draft.Drafter(pipe, steps=4, seed=2).draft("a")
    assert pipe.seeds == [1, 2]
    first, second = pipe.states
    assert not first.equal(second)


def test_prompt_is_passed_as_prompt_not_as_the_leading_image_argument():
    """FluxKontextPipeline.__call__ is (image, prompt, ...).

    Passing the prompt positionally binds it to `image` and leaves `prompt`
    None, which the real pipeline rejects outright. Text-only drafting must
    leave `image` unset.
    """
    pipe = FakePipe()
    draft.Drafter(pipe, steps=4, seed=0).draft("a fox")
    assert pipe.calls[0][0] == "a fox"
    assert pipe.images == [None]


def test_configured_guidance_reaches_the_pipeline():
    """Storing guidance without forwarding it silently ignores the operator's
    setting -- it is a configured knob wired to nothing."""
    pipe = FakePipe()
    draft.Drafter(pipe, steps=4, seed=0, guidance=2.5).draft("a")
    assert pipe.guidances == [2.5]


@pytest.mark.gpu
def test_load_kontext_t2i_returns_a_callable_pipeline():
    pipe = draft.load_kontext_t2i(device="cuda")
    assert callable(pipe)
