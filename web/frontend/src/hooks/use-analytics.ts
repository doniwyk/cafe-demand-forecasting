import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'


export function useABCAnalysis() {
  return useQuery({
    queryKey: ['analytics', 'abc'],
    queryFn: () => api.analytics.abc(),
  })
}

export function useModelMetrics() {
  return useQuery({
    queryKey: ['analytics', 'metrics'],
    queryFn: () => api.analytics.metrics(),
  })
}

export function useAssociationRules(params?: { min_confidence?: number; min_lift?: number }) {
  return useQuery({
    queryKey: ['analytics', 'association-rules', params],
    queryFn: () => api.analytics.associationRules(params),
  })
}
