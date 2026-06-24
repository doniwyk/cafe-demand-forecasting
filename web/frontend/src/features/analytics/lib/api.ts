import { http } from "@/lib/request";
import type { ABCAnalysisResponse, TopSellingItem } from "@/features/analytics/types";

export const analyticsApi = {
  abc() {
    return http.get<ABCAnalysisResponse>("/analytics/abc");
  },
  metrics() {
    return http.get<Record<string, number>>("/analytics/metrics");
  },
  topItems(n?: number) {
    return http.get<TopSellingItem[]>("/analytics/top-items", { n });
  },
};
