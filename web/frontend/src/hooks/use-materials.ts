import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'

export function useDailyNeed(params?: {
  start_date?: string;
  end_date?: string;
  page?: number;
  page_size?: number;
}) {
  return useQuery({
    queryKey: ['materials', 'daily-need', params],
    queryFn: () => api.materials.dailyNeed(params),
  })
}