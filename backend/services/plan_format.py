from enum import Enum


class PlanFormat(str, Enum):
    """
    How a plan was produced, and therefore how it should be shown.

    Carried explicitly rather than inferred from whether descriptions happen to
    be present. A structured plan whose descriptions all came back empty is
    still a structured plan, and inferring the format would render it as plain
    text — a disagreement between what was produced and what is shown, which is
    the failure mode that keeps appearing in this project.
    """

    structured = 'structured'
    text = 'text'
