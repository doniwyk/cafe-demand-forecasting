import { createFileRoute } from '@tanstack/react-router'
import { useState, useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@/components/ui/table'
import { useForecasts, useForecastSummary } from '@/hooks/use-forecasts'
import { useItems } from '@/hooks/use-sales'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'
import { format, parseISO } from 'date-fns'
import { SearchIcon } from 'lucide-react'

export const Route = createFileRoute('/forecasts')({
  component: ForecastsPage,
})

function ForecastsPage() {
  const [search, setSearch] = useState('')
  const [selectedItem, setSelectedItem] = useState<string | null>(null)

  const items = useItems()
  const forecasts = useForecasts({ page_size: 1000 })
  const summary = useForecastSummary()

  const filteredItems = useMemo(() => {
    if (!items.data) return []
    return items.data.filter((i) =>
      i.name.toLowerCase().includes(search.toLowerCase()),
    )
  }, [items.data, search])

  const forecastForItem = useMemo(() => {
    if (!forecasts.data || !selectedItem) return []
    return forecasts.data.data
      .filter((f) => f.item === selectedItem)
      .sort((a, b) => a.date.localeCompare(b.date))
      .map((f) => ({
        date: format(parseISO(f.date), 'MMM dd'),
        predicted: Math.round(f.quantity_sold * 10) / 10,
      }))
  }, [forecasts.data, selectedItem])

  const allItemsChartData = useMemo(() => {
    if (!forecasts.data) return []
    const dailyTotals = new Map<string, number>()
    for (const f of forecasts.data.data) {
      dailyTotals.set(f.date, (dailyTotals.get(f.date) || 0) + f.quantity_sold)
    }
    return Array.from(dailyTotals.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([date, total]) => ({
        date: format(parseISO(date), 'MMM dd'),
        total: Math.round(total),
      }))
  }, [forecasts.data])

  const topForecastItems = useMemo(() => {
    if (!summary.data) return []
    return summary.data.top_items.slice(0, 10)
  }, [summary.data])

  return (
    <div className="flex flex-1 flex-col gap-6 p-4 pt-0">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Sales Forecast</h1>
        <p className="text-sm text-muted-foreground">
          Predicted daily sales for menu items
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Item Selector</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative max-w-sm">
            <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground" />
            <Input
              placeholder="Search items..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-9"
            />
          </div>
          {selectedItem && (
            <div className="mt-3 flex items-center gap-2">
              <Badge variant="secondary">{selectedItem}</Badge>
              <Button variant="ghost" size="sm" onClick={() => setSelectedItem(null)}>
                Clear
              </Button>
            </div>
          )}
          {!selectedItem && filteredItems.length > 0 && (
            <div className="mt-3 flex max-h-40 flex-wrap gap-1.5 overflow-y-auto">
              {filteredItems.slice(0, 50).map((item) => (
                <Badge
                  key={item.name}
                  variant="outline"
                  className="cursor-pointer hover:bg-accent transition-colors"
                  onClick={() => setSelectedItem(item.name)}
                >
                  {item.name}
                </Badge>
              ))}
              {filteredItems.length > 50 && (
                <Badge variant="outline" className="text-muted-foreground">
                  +{filteredItems.length - 50} more...
                </Badge>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>
            {selectedItem ? `Forecast: ${selectedItem}` : 'Total Forecast (All Items)'}
          </CardTitle>
        </CardHeader>
        <CardContent className="min-h-87.5">
          {selectedItem ? (
            forecastForItem.length > 0 ? (
              <ResponsiveContainer width="100%" height={310}>
                <LineChart data={forecastForItem} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
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
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    stroke="var(--chart-2)"
                    strokeWidth={2}
                    dot={false}
                    name="Predicted"
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-77.5 items-center justify-center text-muted-foreground">
                No forecast data for this item
              </div>
            )
          ) : allItemsChartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={310}>
              <LineChart data={allItemsChartData} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
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
                <Line
                  type="monotone"
                  dataKey="total"
                  stroke="var(--chart-1)"
                  strokeWidth={2}
                  dot={false}
                  name="Total Predicted"
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-77.5 items-center justify-center">
              <Skeleton className="h-full w-full" />
            </div>
          )}
        </CardContent>
      </Card>

      {topForecastItems.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Top Items by Forecast Accuracy</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Item</TableHead>
                  <TableHead className="text-right">Actual Qty</TableHead>
                  <TableHead className="text-right">Predicted Qty</TableHead>
                  <TableHead className="text-right">Accuracy</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {topForecastItems.map((item) => (
                  <TableRow key={item.item}>
                    <TableCell className="font-medium">{item.item}</TableCell>
                    <TableCell className="text-right">{Math.round(item.quantity_sold).toLocaleString()}</TableCell>
                    <TableCell className="text-right">{Math.round(item.predicted).toLocaleString()}</TableCell>
                    <TableCell className="text-right">
                      <Badge variant={item.accuracy_pct >= 70 ? 'default' : item.accuracy_pct >= 50 ? 'secondary' : 'destructive'}>
                        {item.accuracy_pct.toFixed(1)}%
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
