import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export function useABCAnalysis(modelType?: string) {
  return useQuery({
    queryKey: ["analytics", "abc", modelType],
    queryFn: () => api.analytics.abc(modelType),
  });
}

export function useModelMetrics(modelType?: string) {
  return useQuery({
    queryKey: ["analytics", "metrics", modelType],
    queryFn: () => api.analytics.metrics(modelType),
  });
}

export function useAssociationRules(
  modelType?: string,
  params?: { min_confidence?: number; min_lift?: number },
) {
  const queryParams = {
    ...params,
    ...(modelType ? { model_type: modelType } : {}),
  };

  return useQuery({
    queryKey: ["analytics", "association-rules", modelType, params],
    queryFn: () => api.analytics.associationRules(queryParams),
  });
}
