import { createFileRoute } from '@tanstack/react-router'
import { useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@/components/ui/table'
import { useABCAnalysis, useModelMetrics, useAssociationRules } from '@/hooks/use-analytics'
import { useForecastSummary } from '@/hooks/use-forecasts'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'

export const Route = createFileRoute('/analytics')({
  component: AnalyticsPage,
})

const CLASS_COLORS: Record<string, string> = {
  A: 'var(--chart-1)',
  B: 'var(--chart-2)',
  C: 'var(--chart-3)',
}

function MetricsGrid({ metrics }: { metrics: Record<string, number> | undefined }) {
  if (!metrics) {
    return <div className="grid grid-cols-2 gap-4 lg:grid-cols-4"><Skeleton className="h-20" /><Skeleton className="h-20" /><Skeleton className="h-20" /><Skeleton className="h-20" /></div>
  }

  const items = [
    { label: 'R-squared', value: metrics['r2']?.toFixed(3) ?? '-', description: 'Higher is better' },
    { label: 'wMAPE', value: `${((metrics['wmape'] ?? 0) * 100).toFixed(1)}%`, description: 'Lower is better' },
    { label: 'MAE', value: metrics['mae']?.toFixed(1) ?? '-', description: 'Mean absolute error' },
    { label: 'Volume Accuracy', value: `${((metrics['volume_accuracy'] ?? 0) * 100).toFixed(1)}%`, description: 'Prediction accuracy' },
  ]

  return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      {items.map((m) => (
        <div key={m.label} className="rounded-lg border p-4">
          <p className="text-xs text-muted-foreground">{m.label}</p>
          <p className="text-xl font-bold mt-1">{m.value}</p>
          <p className="text-xs text-muted-foreground mt-1">{m.description}</p>
        </div>
      ))}
    </div>
  )
}

function AnalyticsPage() {
  const abc = useABCAnalysis()
  const metrics = useModelMetrics()
  const rules = useAssociationRules({ min_confidence: 0.3, min_lift: 1.0 })
  const forecastSummary = useForecastSummary()

  const classBarData = useMemo(() => {
    if (!forecastSummary.data) return []
    return Object.entries(forecastSummary.data.class_metrics).map(([cls, m]) => ({
      class: cls,
      items: m.n_items,
      accuracy: +(m.volume_accuracy * 100).toFixed(1),
    }))
  }, [forecastSummary.data])

  const abcByClass = useMemo(() => {
    if (!abc.data?.classifications) return { A: [], B: [], C: [] }
    const grouped: Record<string, typeof abc.data.classifications> = { A: [], B: [], C: [] }
    for (const item of abc.data.classifications) {
      const cls = item.class_label
      if (grouped[cls]) grouped[cls].push(item)
    }
    return grouped
  }, [abc.data])

  return (
    <div className="flex flex-1 flex-col gap-6 p-4 pt-0">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Analytics</h1>
        <p className="text-sm text-muted-foreground">
          Model performance, ABC classification, and association analysis
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Model Performance</CardTitle>
        </CardHeader>
        <CardContent>
          <MetricsGrid metrics={metrics.data} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>ABC Classification</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 lg:grid-cols-5">
            {classBarData.length > 0 && (
              <div className="lg:col-span-2">
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={classBarData} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="class" tick={{ fontSize: 13, fontWeight: 600 }} stroke="var(--muted-foreground)" />
                    <YAxis tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'var(--popover)',
                        border: '1px solid var(--border)',
                        borderRadius: 'var(--radius)',
                        fontSize: 12,
                      }}
                    />
                    <Bar dataKey="items" name="Items" radius={[4, 4, 0, 0]}>
                      {classBarData.map((d) => (
                        <Cell key={d.class} fill={CLASS_COLORS[d.class] || 'var(--chart-1)'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            <div className="lg:col-span-3">
              {abc.data ? (
                <Tabs defaultValue="A">
                  <TabsList>
                    <TabsTrigger value="A">
                      A-class {abcByClass.A.length > 0 && `(${abcByClass.A.length})`}
                    </TabsTrigger>
                    <TabsTrigger value="B">
                      B-class {abcByClass.B.length > 0 && `(${abcByClass.B.length})`}
                    </TabsTrigger>
                    <TabsTrigger value="C">
                      C-class {abcByClass.C.length > 0 && `(${abcByClass.C.length})`}
                    </TabsTrigger>
                  </TabsList>
                  {(['A', 'B', 'C'] as const).map((cls) => (
                    <TabsContent key={cls} value={cls} className="max-h-65 overflow-y-auto">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Item</TableHead>
                            <TableHead className="text-right">Volume</TableHead>
                            <TableHead className="text-right">Cumulative %</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {abcByClass[cls].length > 0 ? (
                            abcByClass[cls].slice(0, 50).map((item, idx) => (
                              <TableRow key={`${item.item}-${idx}`}>
                                <TableCell className="font-medium">{item.item}</TableCell>
                                <TableCell className="text-right">{Math.round(item.vol).toLocaleString()}</TableCell>
                                <TableCell className="text-right">{(item.pct * 100).toFixed(1)}%</TableCell>
                              </TableRow>
                            ))
                          ) : (
                            <TableRow>
                              <TableCell colSpan={3} className="text-center text-muted-foreground">
                                No items in this class
                              </TableCell>
                            </TableRow>
                          )}
                        </TableBody>
                      </Table>
                    </TabsContent>
                  ))}
                </Tabs>
              ) : (
                <Skeleton className="h-65 w-full" />
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Association Rules</CardTitle>
        </CardHeader>
        <CardContent>
          {rules.data && rules.data.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>If customer buys</TableHead>
                  <TableHead>They also buy</TableHead>
                  <TableHead className="text-right">Support</TableHead>
                  <TableHead className="text-right">Confidence</TableHead>
                  <TableHead className="text-right">Lift</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rules.data.slice(0, 30).map((rule, i) => (
                  <TableRow key={i}>
                    <TableCell>
                      <Badge variant="outline">{rule.antecedents}</Badge>
                    </TableCell>
                    <TableCell>
                      <Badge variant="secondary">{rule.consequents}</Badge>
                    </TableCell>
                    <TableCell className="text-right">{(rule.support * 100).toFixed(1)}%</TableCell>
                    <TableCell className="text-right">{(rule.confidence * 100).toFixed(1)}%</TableCell>
                    <TableCell className="text-right">
                      <Badge variant={rule.lift >= 1.5 ? 'default' : 'secondary'}>
                        {rule.lift.toFixed(2)}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <div className="flex h-50 items-center justify-center text-muted-foreground">
              {rules.isLoading ? <Skeleton className="h-full w-full" /> : 'No strong association rules found'}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
