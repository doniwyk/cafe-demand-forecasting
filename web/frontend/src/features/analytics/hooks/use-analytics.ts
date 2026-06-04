import { useQuery } from "@tanstack/react-query";
import { analyticsApi } from "@/features/analytics/lib/api";
import type { TopSellingItem } from "@/features/analytics/types";

export function useABCAnalysis(modelType?: string) {
  return useQuery({
    queryKey: ["analytics", "abc", modelType],
    queryFn: () => analyticsApi.abc(modelType),
  });
}

export function useModelMetrics(modelType?: string) {
  return useQuery({
    queryKey: ["analytics", "metrics", modelType],
    queryFn: () => analyticsApi.metrics(modelType),
  });
}

export function useTopItems(n = 20) {
  return useQuery<TopSellingItem[]>({
    queryKey: ["analytics", "top-items", n],
    queryFn: () => analyticsApi.topItems(n),
  });
}
