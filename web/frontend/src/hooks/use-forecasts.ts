import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import type { PredictResponse, RetrainResponse } from '@/types'
import { useMutation } from '@tanstack/react-query'

export function useForecasts(params?: { item?: string; start_date?: string; end_date?: string; page?: number; page_size?: number }) {
  return useQuery({
    queryKey: ['forecasts', params],
    queryFn: () => api.forecasts.list(params),
  })
}

export function useForecastSummary() {
  return useQuery({
    queryKey: ['forecasts', 'summary'],
    queryFn: () => api.forecasts.summary(),
  })
}

export function usePredict() {
  return useMutation<PredictResponse, Error, { items: string[]; weeks?: number }>({
    mutationFn: (data) => api.forecasts.predict(data),
  })
}

export function useRetrain() {
  return useMutation<RetrainResponse, Error, void>({
    mutationFn: () => api.forecasts.retrain(),
  })
}

export function useRetrainStatus() {
  return useQuery({
    queryKey: ['forecasts', 'retrain-status'],
    queryFn: () => api.forecasts.retrainStatus(),
    refetchInterval: 5_000,
  })
}
