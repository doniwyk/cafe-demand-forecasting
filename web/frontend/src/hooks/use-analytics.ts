import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'


export function useABCAnalysis() {
  return useQuery({
    queryKey: ['analytics', 'abc'],
    queryFn: () => api.analytics.abc(),
  })
}

export function useModelMetrics(modelType?: string) {
  return useQuery({
    queryKey: ['analytics', 'metrics', modelType],
    queryFn: () => api.analytics.metrics(modelType),
  })
}

export function useAssociationRules(params?: { min_confidence?: number; min_lift?: number }) {
  return useQuery({
    queryKey: ['analytics', 'association-rules', params],
    queryFn: () => api.analytics.associationRules(params),
  })
}
