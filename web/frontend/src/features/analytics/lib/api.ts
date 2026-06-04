import { http } from "@/lib/request";
import type { ABCAnalysisResponse, TopSellingItem } from "@/features/analytics/types";

export const analyticsApi = {
  abc(model_type?: string) {
    return http.get<ABCAnalysisResponse>("/analytics/abc", { model_type });
  },
  metrics(model_type?: string) {
    return http.get<Record<string, number>>("/analytics/metrics", { model_type });
  },
  topItems(n?: number) {
    return http.get<TopSellingItem[]>("/analytics/top-items", { n });
  },
};
