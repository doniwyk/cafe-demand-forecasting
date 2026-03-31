import { createFileRoute } from '@tanstack/react-router'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'
import { useItems, useTopItems } from '@/hooks/use-sales'
import { useForecastSummary } from '@/hooks/use-forecasts'
import { PackageIcon, TargetIcon, BarChart3Icon } from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'
import { useMemo } from 'react'

export const Route = createFileRoute('/')({
  component: DashboardPage,
})

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
    <Card className={cn("flex flex-col flex-1 w-full", className)}>
      <CardHeader className="flex-shrink-0">
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col min-h-0 min-w-0 w-full">
        {children}
      </CardContent>
    </Card>
  )
}

function DashboardPage() {
  const items = useItems()
  const topItems = useTopItems(10)
  const forecastSummary = useForecastSummary()

  const topItemsData = useMemo(() => {
    if (!topItems.data) return []
    return [...topItems.data].slice(0, 10).reverse()
  }, [topItems.data])

  const forecastItemCount = useMemo(() => {
    if (!forecastSummary.data) return 0
    return Object.values(forecastSummary.data.class_metrics).reduce((s, m) => s + m.n_items, 0)
  }, [forecastSummary.data])

  return (
    <div className="flex flex-1 flex-col gap-6 p-4">

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <>
          <KpiCard
            title="Active Items"
            value={items.data?.length ?? '-'}
            description="Menu items tracked"
            icon={<PackageIcon className="size-4" />}
          />
          <KpiCard
            title="Model Accuracy"
            value={forecastSummary.data ? `${forecastSummary.data.global_metrics.volume_accuracy.toFixed(1)}%` : '-'}
            description="Volume accuracy (wMAPE)"
            icon={<TargetIcon className="size-4" />}
          />
          <KpiCard
            title="Items Forecasted"
            value={forecastItemCount || '-'}
            description="Unique items in forecast"
            icon={<BarChart3Icon className="size-4" />}
          />
        </>
      </div>

      <div className="grid gap-4 lg:grid-cols-2 [&>*]:min-w-0">
        <ChartCard title="Top 10 Items by Volume">
          {topItemsData.length > 0 ? (
            <div className="relative h-full w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={topItemsData} margin={{ top: 0, right: 0, left: 0, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis dataKey="item" tick={{ fontSize: 9 }} stroke="var(--muted-foreground)" angle={-35} textAnchor="end" />
                <YAxis type="number" tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" />
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
            </div>
          ) : (
            <div className="flex h-65 items-center justify-center">
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
                    {forecastSummary.data.global_metrics.volume_accuracy.toFixed(1)}%
                  </p>
                </div>
                <div className="rounded-lg border p-3">
                  <p className="text-xs text-muted-foreground">wMAPE</p>
                  <p className="text-xl font-bold">
                    {forecastSummary.data.global_metrics.wmape.toFixed(1)}%
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
                  <p className="text-xl font-bold">{forecastItemCount}</p>
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
                        <span className="font-medium">{m.volume_accuracy.toFixed(1)}% acc</span>
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
