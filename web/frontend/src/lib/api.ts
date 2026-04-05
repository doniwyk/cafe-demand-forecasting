const BASE = '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`)
  }
  return res.json()
}

export const api = {
  sales: {
    daily(params?: { item?: string; start_date?: string; end_date?: string; page?: number; page_size?: number }) {
      const q = new URLSearchParams(params as Record<string, string>)
      return request<import('@/types').DailySalePage>(`/sales/daily?${q}`)
    },
    dailyTotal(params?: { start_date?: string; end_date?: string; page?: number; page_size?: number }) {
      const q = new URLSearchParams(params as Record<string, string>)
      return request<import('@/types').DailyTotalSale[]>(`/sales/daily/total?${q}`)
    },
    dailyCategory(params?: { category?: string; start_date?: string; end_date?: string; page?: number; page_size?: number }) {
      const q = new URLSearchParams(params as Record<string, string>)
      return request<import('@/types').DailyCategorySale[]>(`/sales/daily/category?${q}`)
    },
    items() {
      return request<import('@/types').ItemInfo[]>('/sales/items')
    },
    categories() {
      return request<string[]>('/sales/categories')
    },
  },
  forecasts: {
    list(params?: { item?: string; start_date?: string; end_date?: string; model_type?: string; page?: number; page_size?: number }) {
      const q = new URLSearchParams(params as Record<string, string>)
      return request<import('@/types').ForecastPage>(`/forecasts?${q}`)
    },
    summary(model_type?: string) {
      const q = model_type ? `?model_type=${model_type}` : ''
      return request<import('@/types').ForecastSummary>(`/forecasts/summary${q}`)
    },
    predict(data: { items: string[]; weeks?: number; model_type?: string }) {
      return request<import('@/types').PredictResponse>('/forecasts/predict', {
        method: 'POST',
        body: JSON.stringify(data),
      })
    },
    retrain(model_type: string) {
      return request<import('@/types').RetrainResponse>('/forecasts/retrain', {
        method: 'POST',
        body: JSON.stringify({ model_type }),
      })
    },
    retrainCancel(model_type: string) {
      return request<{ status: string; model_type?: string; message?: string }>('/forecasts/retrain/cancel', {
        method: 'POST',
        body: JSON.stringify({ model_type }),
      })
    },
    retrainStatus() {
      return request<Record<string, { status: string; message: string }>>('/forecasts/retrain/status')
    },
    cleanup() {
      return request<{ deleted_runs: number; deleted_forecasts: number; deleted_class_metrics: number; deleted_top_items: number }>('/forecasts/cleanup', { method: 'POST' })
    },
  },
  materials: {
    daily(params?: { material?: string; start_date?: string; end_date?: string; page?: number; page_size?: number }) {
      const q = new URLSearchParams(params as Record<string, string>)
      return request<import('@/types').MaterialRequirementPage>(`/materials/daily?${q}`)
    },
    forecast(params?: { material?: string; start_date?: string; end_date?: string; page?: number; page_size?: number }) {
      const q = new URLSearchParams(params as Record<string, string>)
      return request<import('@/types').MaterialRequirementPage>(`/materials/forecast?${q}`)
    },
  },
  analytics: {
    abc() {
      return request<import('@/types').ABCAnalysisResponse>('/analytics/abc')
    },
    metrics(model_type?: string) {
      const q = model_type ? `?model_type=${model_type}` : ''
      return request<Record<string, number>>(`/analytics/metrics${q}`)
    },
    topItems(n?: number) {
      return request<import('@/types').TopSellingItem[]>(`/analytics/top-items${n ? `?n=${n}` : ''}`)
    },
    associationRules(params?: { min_confidence?: number; min_lift?: number }) {
      const q = new URLSearchParams(params as Record<string, string>)
      return request<import('@/types').AssociationRule[]>(`/analytics/association-rules?${q}`)
    },
  },
  health() {
    return request<{ status: string }>('/health')
  },
}
