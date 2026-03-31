import { createFileRoute } from '@tanstack/react-router'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'
import { useDailyTotalSales, useItems, useTopItems, useCategorySales } from '@/hooks/use-sales'
import { useForecasts, useForecastSummary } from '@/hooks/use-forecasts'
import { TrendingUpIcon, PackageIcon, TargetIcon, BarChart3Icon } from 'lucide-react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell, Legend,
} from 'recharts'
import { subDays, format } from 'date-fns'
import { useMemo } from 'react'

export const Route = createFileRoute('/')({
  component: DashboardPage,
})

const COLORS = [
  'var(--chart-1)',
  'var(--chart-2)',
  'var(--chart-3)',
  'var(--chart-4)',
  'var(--chart-5)',
  'oklch(0.7 0.15 180)',
  'oklch(0.7 0.15 60)',
  'oklch(0.7 0.15 300)',
]

function KpiCard({ title, value, description, icon }: { title: string; value: string | number; description: string; icon: React.ReactNode }) {
  return (
    <Card>
      <CardHeader>
        <CardDescription>{title}</CardDescription>
        <CardAction icon={icon} />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold tracking-tight">{value}</div>
        <p className="text-xs text-muted-foreground mt-1">{description}</p>
      </CardContent>
    </Card>
  )
}

function CardAction({ icon }: { icon: React.ReactNode }) {
  return (
    <div className="col-start-2 row-span-2 row-start-1 self-start justify-self-end text-muted-foreground">
      {icon}
    </div>
  )
}

function ChartCard({ title, children, className }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <Card className={cn("flex-1", className)}>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent className="min-h-75">
        {children}
      </CardContent>
    </Card>
  )
}

function LoadingCard() {
  return (
    <Card>
      <CardHeader>
        <Skeleton className="h-4 w-24" />
      </CardHeader>
      <CardContent>
        <Skeleton className="h-8 w-32" />
        <Skeleton className="mt-2 h-3 w-48" />
      </CardContent>
    </Card>
  )
}

function DashboardPage() {
  const thirtyDaysAgo = format(subDays(new Date(), 30), 'yyyy-MM-dd')
  const today = format(new Date(), 'yyyy-MM-dd')

  const totalSales = useDailyTotalSales({ start_date: thirtyDaysAgo, end_date: today, page_size: 30 })
  const items = useItems()
  const topItems = useTopItems(10)
  const categorySales = useCategorySales({ start_date: thirtyDaysAgo, end_date: today, page_size: 300 })
  const forecastSummary = useForecastSummary()
  const recentForecasts = useForecasts({ page_size: 100 })

  const kpiData = useMemo(() => {
    if (!totalSales.data) return null
    const last7 = totalSales.data.slice(-7)
    const last30 = totalSales.data
    const totalQty30 = last30.reduce((s, d) => s + d.quantity, 0)
    const totalNetSales30 = last30.reduce((s, d) => s + d.net_sales, 0)
    const avgDaily7 = last7.length > 0 ? last7.reduce((s, d) => s + d.quantity, 0) / last7.length : 0
    return { totalQty30, totalNetSales30, avgDaily7, totalDays: last30.length }
  }, [totalSales.data])

  const trendData = useMemo(() => {
    if (!totalSales.data) return []
    return totalSales.data.map((d) => ({
      date: format(new Date(d.date), 'MMM dd'),
      quantity: Math.round(d.quantity),
      net_sales: Math.round(d.net_sales),
    }))
  }, [totalSales.data])

  const categoryData = useMemo(() => {
    if (!categorySales.data) return []
    const agg = new Map<string, number>()
    for (const d of categorySales.data) {
      agg.set(d.category, (agg.get(d.category) || 0) + d.quantity)
    }
    return Array.from(agg.entries())
      .map(([name, value]) => ({ name, value: Math.round(value) }))
      .sort((a, b) => b.value - a.value)
  }, [categorySales.data])

  const topItemsData = useMemo(() => {
    if (!topItems.data) return []
    return [...topItems.data].slice(0, 10).reverse()
  }, [topItems.data])

  const forecastCount = useMemo(() => {
    if (!recentForecasts.data) return 0
    const uniqueItems = new Set(recentForecasts.data.data.map((f) => f.item))
    return uniqueItems.size
  }, [recentForecasts.data])

  return (
    <div className="flex flex-1 flex-col gap-6 p-4 pt-0">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Dashboard</h1>
          <p className="text-sm text-muted-foreground">
            Overview of sales performance and forecasts
          </p>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {kpiData ? (
          <>
            <KpiCard
              title="Total Sales (30d)"
              value={kpiData.totalQty30.toLocaleString()}
              description={`Net revenue: Rp ${(kpiData.totalNetSales30 / 1_000_000).toFixed(1)}M`}
              icon={<TrendingUpIcon className="size-4" />}
            />
            <KpiCard
              title="Avg Daily Sales"
              value={Math.round(kpiData.avgDaily7).toLocaleString()}
              description="Average over last 7 days"
              icon={<BarChart3Icon className="size-4" />}
            />
            <KpiCard
              title="Active Items"
              value={items.data?.length ?? '-'}
              description="Menu items tracked"
              icon={<PackageIcon className="size-4" />}
            />
            <KpiCard
              title="Model Accuracy"
              value={forecastSummary.data ? `${(forecastSummary.data.global_metrics.volume_accuracy * 100).toFixed(1)}%` : '-'}
              description="Volume accuracy (wMAPE)"
              icon={<TargetIcon className="size-4" />}
            />
          </>
        ) : (
          <>
            <LoadingCard />
            <LoadingCard />
            <LoadingCard />
            <LoadingCard />
          </>
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-7">
        <ChartCard title="Sales Trend (30 Days)" className="lg:col-span-4">
          {trendData.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={trendData} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="salesGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--chart-1)" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="var(--chart-1)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" />
                <YAxis tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'var(--popover)',
                    border: '1px solid var(--border)',
                    borderRadius: 'var(--radius)',
                    fontSize: 12,
                  }}
                />
                <Area type="monotone" dataKey="quantity" stroke="var(--chart-1)" fill="url(#salesGradient)" strokeWidth={2} name="Quantity" />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-65 items-center justify-center">
              <Skeleton className="h-full w-full" />
            </div>
          )}
        </ChartCard>

        <ChartCard title="Top 10 Items" className="lg:col-span-3">
          {topItemsData.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={topItemsData} layout="vertical" margin={{ top: 0, right: 20, left: 80, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" />
                <YAxis type="category" dataKey="item" tick={{ fontSize: 10 }} stroke="var(--muted-foreground)" width={75} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'var(--popover)',
                    border: '1px solid var(--border)',
                    borderRadius: 'var(--radius)',
                    fontSize: 12,
                  }}
                />
                <Bar dataKey="total_quantity" fill="var(--chart-2)" radius={[0, 4, 4, 0]} name="Total Qty" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-65 items-center justify-center">
              <Skeleton className="h-full w-full" />
            </div>
          )}
        </ChartCard>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <ChartCard title="Category Breakdown">
          {categoryData.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  data={categoryData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={2}
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${((percent ?? 0) * 100).toFixed(0)}%`}
                  labelLine={false}
                  fontSize={11}
                >
                  {categoryData.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'var(--popover)',
                    border: '1px solid var(--border)',
                    borderRadius: 'var(--radius)',
                    fontSize: 12,
                  }}
                />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-70 items-center justify-center">
              <Skeleton className="h-full w-full" />
            </div>
          )}
        </ChartCard>

        <ChartCard title="Forecast Summary">
          {forecastSummary.data ? (
            <div className="flex flex-col gap-4 pt-2">
              <div className="grid grid-cols-2 gap-4">
                <div className="rounded-lg border p-3">
                  <p className="text-xs text-muted-foreground">Volume Accuracy</p>
                  <p className="text-xl font-bold">
                    {(forecastSummary.data.global_metrics.volume_accuracy * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="rounded-lg border p-3">
                  <p className="text-xs text-muted-foreground">wMAPE</p>
                  <p className="text-xl font-bold">
                    {(forecastSummary.data.global_metrics.wmape * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="rounded-lg border p-3">
                  <p className="text-xs text-muted-foreground">R-squared</p>
                  <p className="text-xl font-bold">
                    {forecastSummary.data.global_metrics.r2.toFixed(3)}
                  </p>
                </div>
                <div className="rounded-lg border p-3">
                  <p className="text-xs text-muted-foreground">Items Forecasted</p>
                  <p className="text-xl font-bold">{forecastCount}</p>
                </div>
              </div>

              <div>
                <h4 className="text-sm font-medium mb-2">Accuracy by Class</h4>
                <div className="space-y-2">
                  {Object.entries(forecastSummary.data.class_metrics).map(([cls, m]) => (
                    <div key={cls} className="flex items-center justify-between rounded-lg border p-2">
                      <span className="text-sm font-medium">{cls}</span>
                      <div className="flex gap-4 text-sm">
                        <span className="text-muted-foreground">{m.n_items} items</span>
                        <span className="font-medium">{(m.volume_accuracy * 100).toFixed(1)}% acc</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="flex h-70 items-center justify-center">
              <Skeleton className="h-full w-full" />
            </div>
          )}
        </ChartCard>
      </div>
    </div>
  )
}
