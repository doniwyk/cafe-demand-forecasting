import { createFileRoute } from '@tanstack/react-router'
import { useState, useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@/components/ui/table'
import { useDailyNeed } from '@/hooks/use-materials'
import { format, parseISO } from 'date-fns'
import { SearchIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

export const Route = createFileRoute('/materials/daily-need')({
  component: DailyNeedPage,
})

function DailyNeedPage() {
  const [search, setSearch] = useState('')
  const [selectedMaterial, setSelectedMaterial] = useState<string | null>(null)
  const [selectedDate, setSelectedDate] = useState<string | null>('2025-10-10')
  const { t } = useTranslation()

  const dailyNeed = useDailyNeed({ page_size: 1000 })

  const filteredMaterials = useMemo(() => {
    if (!dailyNeed.data) return []
    const materials = [...new Set(dailyNeed.data.data.map(d => d.raw_material))]
    return materials.filter(m => m.toLowerCase().includes(search.toLowerCase()))
  }, [dailyNeed.data, search])

  const filteredData = useMemo(() => {
    if (!dailyNeed.data) return []
    let data = dailyNeed.data.data
    
    if (selectedMaterial) {
      data = data.filter(d => d.raw_material === selectedMaterial)
    }
    if (selectedDate) {
      data = data.filter(d => d.date === selectedDate)
    }
    
    return data.sort((a, b) => {
      const dateCmp = a.date.localeCompare(b.date)
      if (dateCmp !== 0) return dateCmp
      return a.raw_material.localeCompare(b.raw_material)
    })
  }, [dailyNeed.data, selectedMaterial, selectedDate])

  const chartData = useMemo(() => {
    if (!dailyNeed.data) return []
    const totals = new Map<string, number>()
    
    for (const d of dailyNeed.data.data) {
      const key = d.date
      totals.set(key, (totals.get(key) || 0) + d.quantity_required)
    }
    
    return Array.from(totals.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([date, total]) => ({
        date: format(parseISO(date), 'MMM dd'),
        total: Math.round(total),
      }))
  }, [dailyNeed.data])

  return (
    <div className="flex flex-1 flex-col gap-6 p-4">

      <Card data-tour="material-filter">
        <CardHeader>
          <CardTitle>{t("materials.materialFilter")}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4 flex-wrap">
            <div className="relative max-w-sm flex-1 min-w-48">
              <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground" />
              <Input
                placeholder={t("materials.searchMaterials")}
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-9"
              />
            </div>
            <div className="flex gap-2 flex-wrap">
              {filteredMaterials.slice(0, 15).map((mat) => (
                <Badge
                  key={mat}
                  variant={selectedMaterial === mat ? 'default' : 'outline'}
                  className="cursor-pointer hover:bg-accent"
                  onClick={() => setSelectedMaterial(selectedMaterial === mat ? null : mat)}
                >
                  {mat}
                </Badge>
              ))}
              {filteredMaterials.length > 15 && (
                <Badge variant="outline" className="text-muted-foreground">
                  +{filteredMaterials.length - 15} more
                </Badge>
              )}
            </div>
          </div>
          {selectedMaterial && (
            <div className="mt-3 flex items-center gap-2">
              <Badge variant="secondary">{t("materials.material", { name: selectedMaterial })}</Badge>
              <Button variant="ghost" size="sm" onClick={() => setSelectedMaterial(null)}>
                {t("common.clear")}
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      <Card data-tour="daily-requirements">
        <CardHeader>
          <CardTitle>{t("materials.dailyMaterialRequirements")}</CardTitle>
        </CardHeader>
        <CardContent>
          {dailyNeed.isLoading ? (
            <div className="space-y-2">
              {[...Array(5)].map((_, i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : filteredData.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{t("materials.date")}</TableHead>
                  <TableHead>{t("materials.material")}</TableHead>
                  <TableHead className="text-right">{t("materials.quantityRequired")}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredData.slice(0, 50).map((row, idx) => (
                  <TableRow 
                    key={`${row.date}-${row.raw_material}-${idx}`}
                    className={selectedDate === row.date ? 'bg-accent' : ''}
                    onClick={() => setSelectedDate(selectedDate === row.date ? null : row.date)}
                  >
                    <TableCell className="font-medium">
                      {format(parseISO(row.date), 'MMM dd, yyyy')}
                    </TableCell>
                    <TableCell>{row.raw_material}</TableCell>
                    <TableCell className="text-right">
                      {Math.round(row.quantity_required * 100) / 100}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              {t("materials.noMaterialData")}
            </div>
          )}
        </CardContent>
      </Card>

      {dailyNeed.data && dailyNeed.data.data.length > 0 && (
        <Card data-tour="requirements-trend">
          <CardHeader>
            <CardTitle>{t("materials.totalDailyTrend")}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-sm text-muted-foreground mb-4">
              {t("materials.totalDailyDesc", {
                total: Math.round(dailyNeed.data.data.reduce((acc, d) => acc + d.quantity_required, 0)).toLocaleString(),
                records: dailyNeed.data.total,
              })}
            </div>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{t("materials.date")}</TableHead>
                  <TableHead className="text-right">{t("materials.totalQuantity")}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {chartData.slice(0, 14).map((row) => (
                  <TableRow 
                    key={row.date}
                    className={selectedDate === row.date ? 'bg-accent' : ''}
                    onClick={() => setSelectedDate(selectedDate === row.date ? null : row.date)}
                  >
                    <TableCell className="font-medium">{row.date}</TableCell>
                    <TableCell className="text-right">{row.total.toLocaleString()}</TableCell>
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
