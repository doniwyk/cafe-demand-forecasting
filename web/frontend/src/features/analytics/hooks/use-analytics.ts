import { useQuery } from "@tanstack/react-query";
import { analyticsApi } from "@/features/analytics/lib/api";


export function useABCAnalysis() {
  return useQuery({
    queryKey: ["analytics", "abc"],
    queryFn: () => analyticsApi.abc(),
  });
}

export function useModelMetrics() {
  return useQuery({
    queryKey: ["analytics", "metrics"],
    queryFn: () => analyticsApi.metrics(),
  });
}
