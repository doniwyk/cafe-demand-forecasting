import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import type { TopSellingItem } from '@/types'

export function useDailySales(params?: { item?: string; start_date?: string; end_date?: string; page?: number; page_size?: number }) {
  return useQuery({
    queryKey: ['sales', 'daily', params],
    queryFn: () => api.sales.daily(params),
  })
}

export function useDailyTotalSales(params?: { start_date?: string; end_date?: string; page?: number; page_size?: number }) {
  return useQuery({
    queryKey: ['sales', 'total', params],
    queryFn: () => api.sales.dailyTotal(params),
  })
}

export function useCategorySales(params?: { category?: string; start_date?: string; end_date?: string; page?: number; page_size?: number }) {
  return useQuery({
    queryKey: ['sales', 'category', params],
    queryFn: () => api.sales.dailyCategory(params),
  })
}

export function useItems() {
  return useQuery({
    queryKey: ['sales', 'items'],
    queryFn: () => api.sales.items(),
  })
}

export function useCategories() {
  return useQuery({
    queryKey: ['sales', 'categories'],
    queryFn: () => api.sales.categories(),
  })
}

export function useTopItems(n = 20) {
  return useQuery<TopSellingItem[]>({
    queryKey: ['analytics', 'top-items', n],
    queryFn: () => api.analytics.topItems(n),
  })
}
