import { useQuery } from "@tanstack/react-query";
import { analyticsApi } from "@/features/analytics/lib/api";


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


