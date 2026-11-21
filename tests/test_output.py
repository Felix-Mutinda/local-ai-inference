from local_ai_inference.server.output import StopMatcher


def stream_through(matcher: StopMatcher, pieces: list[str]) -> tuple[str, bool]:
    out = []
    for piece in pieces:
        safe, hit = matcher.push(piece)
        out.append(safe)
        if hit:
            return "".join(out), True
    out.append(matcher.flush())
    return "".join(out), False


def test_no_stops_passes_text_through() -> None:
    assert stream_through(StopMatcher(), ["a", "bc", "d"]) == ("abcd", False)


def test_stop_split_across_pieces_is_never_emitted() -> None:
    text, hit = stream_through(StopMatcher(["END"]), ["hello E", "N", "D more"])
    assert (text, hit) == ("hello ", True)


def test_partial_stop_prefix_is_released_when_it_does_not_complete() -> None:
    assert stream_through(StopMatcher(["END"]), ["the EN", "gine"]) == ("the ENgine", False)


def test_earliest_of_several_stops_wins() -> None:
    assert stream_through(StopMatcher(["two", "one"]), ["zero one two"]) == ("zero ", True)
