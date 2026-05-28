import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { useModelType } from '@/contexts/model-context'

export function useDailyNeed(params?: {
  material?: string;
  start_date?: string;
  end_date?: string;
  page?: number;
  page_size?: number;
}) {
  const { modelType } = useModelType()
  const queryParams = { ...params, model_type: modelType }
  return useQuery({
    queryKey: ['materials', 'daily-need', queryParams],
    queryFn: () => api.materials.forecast(queryParams),
  })
}