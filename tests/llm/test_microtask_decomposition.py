from pb.llm.drafts import LearningPlanBlockDraft


def test_learning_plan_block_supports_sub_index():
    block = LearningPlanBlockDraft(
        branch="study",
        subject_scope="Multivariable calculus",
        duration_minutes=30,
        sub_index="2a",
    )
    assert block.sub_index == "2a"


def test_sub_index_defaults_to_none():
    block = LearningPlanBlockDraft(
        branch="study",
        subject_scope="German vocab",
        duration_minutes=45,
    )
    assert block.sub_index is None
