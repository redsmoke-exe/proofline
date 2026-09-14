from __future__ import annotations

from cv_agent.schemas import PipelineSnapshot
from cv_agent.strategy import build_resume_strategy


def test_strategy_selects_us_letter_and_experienced_content_order(
    snapshot: PipelineSnapshot,
) -> None:
    strategy = build_resume_strategy(
        snapshot.master_profile,
        snapshot.job_requirements,
    )

    assert strategy.career_stage == "experienced"
    assert strategy.target_market == "us"
    assert strategy.page_size == "Letter"
    assert strategy.include_summary is True
    assert strategy.section_order[:3] == ["summary", "skills", "experience"]
    assert strategy.target_pages in {1, 2}
    assert all(question.endswith("?") for question in strategy.evidence_gap_questions)


def test_strategy_prioritizes_education_for_student_profile(
    snapshot: PipelineSnapshot,
) -> None:
    student = snapshot.master_profile.model_copy(
        update={
            "contact": snapshot.master_profile.contact.model_copy(
                update={"location": "Bengaluru, India"}
            ),
            "summary_facts": [],
            "experience": [],
        }
    )

    strategy = build_resume_strategy(student, snapshot.job_requirements)

    assert strategy.career_stage == "student"
    assert strategy.target_market == "india"
    assert strategy.page_size == "A4"
    assert strategy.target_pages == 1
    assert strategy.include_summary is False
    assert strategy.section_order[:2] == ["education", "skills"]
    assert "summary" not in strategy.section_order
    assert "experience" not in strategy.section_order


def test_market_detection_does_not_treat_any_two_letter_region_as_us(
    snapshot: PipelineSnapshot,
) -> None:
    canadian_profile = snapshot.master_profile.model_copy(
        update={
            "contact": snapshot.master_profile.contact.model_copy(
                update={"location": "Toronto, ON"}
            )
        }
    )
    lowercase_us_profile = snapshot.master_profile.model_copy(
        update={
            "contact": snapshot.master_profile.contact.model_copy(
                update={"location": "Austin, tx"}
            )
        }
    )

    canadian = build_resume_strategy(canadian_profile, snapshot.job_requirements)
    us = build_resume_strategy(lowercase_us_profile, snapshot.job_requirements)

    assert (canadian.target_market, canadian.page_size) == ("global", "A4")
    assert (us.target_market, us.page_size) == ("us", "Letter")
