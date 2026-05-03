import { createFileRoute } from '@tanstack/react-router'
import { useState, useMemo, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@/components/ui/table'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Command, CommandInput, CommandList, CommandEmpty, CommandGroup, CommandItem } from '@/components/ui/command'
import { useDailyNeed } from '@/hooks/use-materials'
import { format, parseISO } from 'date-fns'
import { Calendar } from '@/components/ui/calendar'
import { ChevronsUpDownIcon, DownloadIcon, CalendarIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

export const Route = createFileRoute('/materials/daily-need')({
  component: DailyNeedPage,
})

function DailyNeedPage() {
  const [selectedMaterial, setSelectedMaterial] = useState<string | null>(null)
  const [dateRange, setDateRange] = useState<{ from?: Date; to?: Date }>({})
  const [materialOpen, setMaterialOpen] = useState(false)
  const [dateOpen, setDateOpen] = useState(false)
  const { t } = useTranslation()

  const dateParams = useMemo(() => {
    const p: { start_date?: string; end_date?: string; page_size: number } = { page_size: 100 }
    if (dateRange.from) p.start_date = format(dateRange.from, 'yyyy-MM-dd')
    if (dateRange.to) p.end_date = format(dateRange.to, 'yyyy-MM-dd')
    return p
  }, [dateRange.from, dateRange.to])

  const dailyNeed = useDailyNeed(dateParams)

  const materials = useMemo(() => {
    if (!dailyNeed.data) return []
    return [...new Set(dailyNeed.data.data.map(d => d.raw_material))].sort()
  }, [dailyNeed.data])

  const filteredData = useMemo(() => {
    if (!dailyNeed.data) return []
    let data = dailyNeed.data.data
    
    if (selectedMaterial) {
      data = data.filter(d => d.raw_material === selectedMaterial)
    }
    
    return data.sort((a, b) => {
      const dateCmp = a.date.localeCompare(b.date)
      if (dateCmp !== 0) return dateCmp
      return a.raw_material.localeCompare(b.raw_material)
    })
  }, [dailyNeed.data, selectedMaterial])

  const exportCsv = useCallback(() => {
    const headers = ['Date', 'Material', 'Quantity Required']
    const rows = filteredData.map(d => [
      d.date,
      d.raw_material,
      d.quantity_required,
    ])
    const csv = [headers, ...rows].map(r => r.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `material-requirements-${new Date().toISOString().slice(0, 10)}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }, [filteredData])

  return (
    <div className="flex flex-1 flex-col gap-6 p-4">

      <Card data-tour="material-filter">
        <CardHeader>
          <CardTitle>{t("materials.materialFilter")}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4 flex-wrap items-end">
            <div className="flex flex-col gap-1.5">
              <label className="text-sm font-medium">Material</label>
              <Popover open={materialOpen} onOpenChange={setMaterialOpen}>
                <PopoverTrigger
                  render={<Button variant="outline" role="combobox" aria-expanded={open} className="w-[280px] justify-between" />}
                >
                  {selectedMaterial || t("materials.selectMaterial")}
                  <ChevronsUpDownIcon className="ml-auto size-4 shrink-0 opacity-50" />
                </PopoverTrigger>
                <PopoverContent className="w-[280px] p-0" align="start">
                  <Command>
                    <CommandInput placeholder={t("materials.searchMaterials")} />
                    <CommandList>
                      <CommandEmpty>{t("materials.noMaterialsFound")}</CommandEmpty>
                      <CommandGroup>
                        {materials.map((mat) => (
                          <CommandItem
                            key={mat}
                            value={mat}
                            onSelect={(value) => {
                              setSelectedMaterial(value === selectedMaterial ? null : value)
                              setOpen(false)
                            }}
                          >
                            {mat}
                          </CommandItem>
                        ))}
                      </CommandGroup>
                    </CommandList>
                  </Command>
                </PopoverContent>
              </Popover>
            </div>
            <div className="flex flex-col gap-1.5">
              <label className="text-sm font-medium">{t("materials.dateRange")}</label>
              <Popover open={dateOpen} onOpenChange={setDateOpen}>
                <PopoverTrigger
                  render={
                    <Button variant="outline" className="w-[280px] justify-start text-left font-normal">
                      <CalendarIcon className="mr-2 size-4 shrink-0" />
                      {dateRange.from ? (
                        dateRange.to ? (
                          <>{format(dateRange.from, 'MMM dd, yyyy')} — {format(dateRange.to, 'MMM dd, yyyy')}</>
                        ) : (
                          format(dateRange.from, 'MMM dd, yyyy')
                        )
                      ) : (
                        <span className="text-muted-foreground">{t("materials.selectDateRange")}</span>
                      )}
                    </Button>
                  }
                />
                <PopoverContent className="w-auto p-0" align="start">
                  <Calendar
                    mode="range"
                    selected={dateRange}
                    onSelect={(range) => setDateRange(range ?? {})}
                    numberOfMonths={2}
                  />
                </PopoverContent>
              </Popover>
            </div>
            {(selectedMaterial || dateRange.from || dateRange.to) && (
              <Button variant="ghost" size="sm" onClick={() => { setSelectedMaterial(null); setDateRange({}) }}>
                {t("common.clear")}
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      <Card data-tour="daily-requirements">
        <CardHeader>
          <CardTitle>{t("materials.dailyMaterialRequirements")}</CardTitle>
          <Button variant="outline" size="sm" onClick={exportCsv} disabled={filteredData.length === 0}>
            <DownloadIcon className="size-4" />
            Export Excel
          </Button>
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
                  <TableRow key={`${row.date}-${row.raw_material}-${idx}`}>
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


    </div>
  )
}
