from glyphstack import target, template


def test_a_twin_takes_its_low_form_on_the_baseline_and_its_high_form_above_it():
    target.use(target.DEFAULT)
    em = target.get()["templates"]["px_per_em"] / template.UP
    # letters 0.5 em tall on a baseline at y 1000, and a mark beside them
    letters = [dict(bbox=(1000 - int(0.5 * em), 1000, x, x + 60)) for x in range(100, 700, 80)]
    low = dict(bbox=(990, 1030, 720, 760))
    high = dict(bbox=(1000 - int(0.75 * em), 1000 - int(0.45 * em), 720, 760))
    assert template.place_twin(low, letters + [low], 1.0, "„”") == "„"
    assert template.place_twin(high, letters + [high], 1.0, "„”") == "”"
    # with no letters beside it a mark is left unlabelled
    assert template.place_twin(low, [low], 1.0, "„”") is None


def test_a_low_mark_takes_the_face_of_the_letters_on_its_line():
    target.use(target.DEFAULT)
    em = target.get()["templates"]["px_per_em"] / template.UP
    letters = [dict(bbox=(1000 - int(0.7 * em), 1000, x, x + 60), cx=x + 30, ch="i:A") for x in range(100, 400, 80)]
    above = [dict(bbox=(1000 - int(1.9 * em), 1000 - int(1.2 * em), 420, 480), cx=450, ch="A")]
    under = dict(bbox=(1002, 1012, 410, 490), cx=450, ch="_", scale=1.0)
    assert template.resolve_faces([under], letters + above)[0]["ch"] == "i:_"
