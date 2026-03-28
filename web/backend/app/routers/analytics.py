from fastapi import APIRouter, Query

from app.models.analytics import ABCAnalysisResponse, AssociationRule
from app.services import analytics_service

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/abc", response_model=ABCAnalysisResponse)
def get_abc_analysis():
    return analytics_service.get_abc_analysis()


@router.get("/metrics")
def get_metrics():
    return analytics_service.get_metrics()


@router.get("/top-items")
def get_top_items(n: int = Query(20, ge=1, le=100)):
    return analytics_service.get_top_items(n)


@router.get("/association-rules", response_model=list[AssociationRule])
def get_association_rules(
    min_confidence: float = Query(0.3, ge=0, le=1),
    min_lift: float = Query(1.0, ge=0),
):
    return analytics_service.get_association_rules(min_confidence, min_lift)
