import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import type { PredictResponse, RetrainResponse } from '@/types'
import { useMutation } from '@tanstack/react-query'

export function useForecasts(params?: { item?: string; start_date?: string; end_date?: string; model_type?: string; page?: number; page_size?: number }) {
  return useQuery({
    queryKey: ['forecasts', params],
    queryFn: () => api.forecasts.list(params),
  })
}

export function useForecastSummary(modelType?: string) {
  return useQuery({
    queryKey: ['forecasts', 'summary', modelType],
    queryFn: () => api.forecasts.summary(modelType),
  })
}

export function usePredict() {
  return useMutation<PredictResponse, Error, { items: string[]; weeks?: number; model_type?: string }>({
    mutationFn: (data) => api.forecasts.predict(data),
  })
}

export function useRetrain() {
  return useMutation<RetrainResponse, Error, string>({
    mutationFn: (modelType) => api.forecasts.retrain(modelType),
  })
}

export function useRetrainCancel() {
  return useMutation<{ status: string; model_type?: string }, Error, string>({
    mutationFn: (modelType) => api.forecasts.retrainCancel(modelType),
  })
}

export function useRetrainStatus() {
  return useQuery({
    queryKey: ['forecasts', 'retrain-status'],
    queryFn: () => api.forecasts.retrainStatus(),
    refetchInterval: 5_000,
  })
}

export function useCleanup() {
  return useMutation<{
    deleted_runs: number
    deleted_forecasts: number
    deleted_class_metrics: number
    deleted_top_items: number
  }, Error, void>({
    mutationFn: () => api.forecasts.cleanup(),
  })
}
