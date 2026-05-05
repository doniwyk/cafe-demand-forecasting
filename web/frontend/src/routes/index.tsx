import { createFileRoute } from '@tanstack/react-router'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'
import { useTopItems } from '@/hooks/use-sales'
import { useForecastSummary } from '@/hooks/use-forecasts'
import { useModelType } from '@/contexts/model-context'
import { PackageIcon, TargetIcon, BarChart3Icon } from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'
import { useMemo } from 'react'
import { useTranslation } from 'react-i18next'

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

function ChartCard({ title, children, className, ...props }: { title: string; children: React.ReactNode; className?: string } & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <Card className={cn("flex flex-col flex-1 w-full", className)} {...props}>
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
  const { modelType } = useModelType()
  const topItems = useTopItems(10)
  const forecastSummary = useForecastSummary(modelType)
  const { t } = useTranslation()

  const topItemsData = useMemo(() => {
    if (!topItems.data) return []
    return [...topItems.data].slice(0, 10).reverse()
  }, [topItems.data])

  const forecastItemCount = useMemo(() => {
    if (!forecastSummary.data) return 0
    return Object.values(forecastSummary.data.class_metrics).reduce((s, m) => s + m.n_items, 0)
  }, [forecastSummary.data])

  const activeItemCount = useMemo(() => {
    if (!forecastSummary.data) return 0
    return forecastItemCount
  }, [forecastSummary.data, forecastItemCount])

  return (
    <div className="flex flex-1 flex-col gap-6 p-4">

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3" data-tour="kpi-cards">
        <>
          <KpiCard
            title={t("dashboard.activeItems")}
            value={activeItemCount || '-'}
            description={t("dashboard.activeItemsDesc")}
            icon={<PackageIcon className="size-4" />}
          />
          <KpiCard
            title={t("dashboard.modelAccuracy")}
            value={forecastSummary.data ? `${forecastSummary.data.global_metrics.median_period_accuracy.toFixed(1)}%` : '-'}
            description={t("dashboard.medianPeriodAccuracyDesc")}
            icon={<TargetIcon className="size-4" />}
          />
          <KpiCard
            title={t("dashboard.itemsForecasted")}
            value={forecastItemCount || '-'}
            description={t("dashboard.itemsForecastedDesc")}
            icon={<BarChart3Icon className="size-4" />}
          />
        </>
      </div>

      <div className="grid gap-4 lg:grid-cols-2 [&>*]:min-w-0">
        <ChartCard title={t("dashboard.topItemsByVolume")} data-tour="top-items-chart">
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
                  <Bar dataKey="total_quantity" fill="var(--chart-2)" radius={[0, 4, 4, 0]} name={t("dashboard.totalQty")} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="flex h-65 items-center justify-center">
              <Skeleton className="h-full w-full" />
            </div>
          )}
        </ChartCard>

        <ChartCard title={t("dashboard.forecastSummary")} data-tour="forecast-summary">
          {forecastSummary.data ? (
            <div className="flex flex-col gap-4 pt-2">
              <div className="grid grid-cols-2 gap-4">
                <div className="rounded-lg border p-3">
                  <p className="text-xs text-muted-foreground">{t("dashboard.medianPeriodAccuracy")}</p>
                  <p className="text-xl font-bold">
                    {forecastSummary.data.global_metrics.median_period_accuracy.toFixed(1)}%
                  </p>
                </div>
                <div className="rounded-lg border p-3">
                  <p className="text-xs text-muted-foreground">{t("dashboard.wmape")}</p>
                  <p className="text-xl font-bold">
                    {forecastSummary.data.global_metrics.wmape.toFixed(1)}%
                  </p>
                </div>
                <div className="rounded-lg border p-3">
                  <p className="text-xs text-muted-foreground">{t("dashboard.rSquared")}</p>
                  <p className="text-xl font-bold">
                    {forecastSummary.data.global_metrics.r2.toFixed(3)}
                  </p>
                </div>
                <div className="rounded-lg border p-3">
                  <p className="text-xs text-muted-foreground">{t("dashboard.itemsForecasted")}</p>
                  <p className="text-xl font-bold">{forecastItemCount}</p>
                </div>
              </div>

              <div>
                <h4 className="text-sm font-medium mb-2">{t("dashboard.accuracyByClass")}</h4>
                <div className="space-y-2">
                  {Object.entries(forecastSummary.data.class_metrics).map(([cls, m]) => (
                    <div key={cls} className="flex items-center justify-between rounded-lg border p-2">
                      <span className="text-sm font-medium">{cls}</span>
                      <div className="flex gap-4 text-sm">
                        <span className="text-muted-foreground">{m.n_items} {t("dashboard.items")}</span>
                        <span className="font-medium">{m.median_period_accuracy.toFixed(1)}% {t("dashboard.accuracy")}</span>
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
