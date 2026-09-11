"""Literal position-update coefficient stated in Appendix L."""

from pep_compass.optimization.components.walkers.strategies.sorbes.position_update.main import MainPositionUpdate


class ArticlePositionUpdate(MainPositionUpdate):
    """Apply the article's ``epsilon*v - epsilon**2*Gamma[v,v]`` form."""

    # REMARK: main uses coefficient 0.5, while Appendix L prints coefficient 1.
    # TODO: Confirm whether its Gamma term equals the acceleration used by main.
    acceleration_coefficient = 1.0
