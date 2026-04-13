import { createContext, useContext, useState, useCallback, useRef, useEffect, useMemo, type ReactNode } from 'react'
import { driver, type Config, type DriveStep } from 'driver.js'
import 'driver.js/dist/driver.css'
import i18n from '@/lib/i18n'

const STORAGE_KEY = 'cafe-tour-completed'

function getCompleted(): Record<string, boolean> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : {}
  } catch {
    return {}
  }
}

function markCompleted(page: string) {
  const completed = getCompleted()
  completed[page] = true
  localStorage.setItem(STORAGE_KEY, JSON.stringify(completed))
}

function hasCompleted(page: string): boolean {
  return !!getCompleted()[page]
}

function resetAll() {
  localStorage.removeItem(STORAGE_KEY)
}

interface StepDef {
  titleKey: string
  descKey: string
  element: string
  side?: 'left' | 'right' | 'top' | 'bottom'
  align?: 'start' | 'center' | 'end'
}

const TOUR_STEPS: Record<string, StepDef[]> = {
  '/': [
    { titleKey: 'tour.sidebar.title', descKey: 'tour.sidebar.desc', element: '[data-tour="sidebar"]', side: 'right', align: 'start' },
    { titleKey: 'tour.modelSelector.title', descKey: 'tour.modelSelector.desc', element: '[data-tour="model-selector"]', side: 'bottom', align: 'end' },
    { titleKey: 'tour.kpiCards.title', descKey: 'tour.kpiCards.desc', element: '[data-tour="kpi-cards"]', side: 'bottom', align: 'start' },
    { titleKey: 'tour.topItemsChart.title', descKey: 'tour.topItemsChart.desc', element: '[data-tour="top-items-chart"]', side: 'left', align: 'start' },
    { titleKey: 'tour.forecastSummary.title', descKey: 'tour.forecastSummary.desc', element: '[data-tour="forecast-summary"]', side: 'left', align: 'start' },
  ],
  '/analytics': [
    { titleKey: 'tour.modelPerformance.title', descKey: 'tour.modelPerformance.desc', element: '[data-tour="model-performance"]', side: 'bottom', align: 'start' },
    { titleKey: 'tour.abcClassification.title', descKey: 'tour.abcClassification.desc', element: '[data-tour="abc-classification"]', side: 'bottom', align: 'start' },
    { titleKey: 'tour.associationRules.title', descKey: 'tour.associationRules.desc', element: '[data-tour="association-rules"]', side: 'top', align: 'start' },
  ],
  '/forecasts': [
    { titleKey: 'tour.itemSelector.title', descKey: 'tour.itemSelector.desc', element: '[data-tour="item-selector"]', side: 'bottom', align: 'start' },
    { titleKey: 'tour.forecastChart.title', descKey: 'tour.forecastChart.desc', element: '[data-tour="forecast-chart"]', side: 'top', align: 'start' },
    { titleKey: 'tour.topAccuracyTable.title', descKey: 'tour.topAccuracyTable.desc', element: '[data-tour="top-accuracy-table"]', side: 'top', align: 'start' },
  ],
  '/materials/daily-need': [
    { titleKey: 'tour.materialFilter.title', descKey: 'tour.materialFilter.desc', element: '[data-tour="material-filter"]', side: 'bottom', align: 'start' },
    { titleKey: 'tour.dailyRequirements.title', descKey: 'tour.dailyRequirements.desc', element: '[data-tour="daily-requirements"]', side: 'top', align: 'start' },
    { titleKey: 'tour.requirementsTrend.title', descKey: 'tour.requirementsTrend.desc', element: '[data-tour="requirements-trend"]', side: 'top', align: 'start' },
  ],
  '/settings': [
    { titleKey: 'tour.modelTraining.title', descKey: 'tour.modelTraining.desc', element: '[data-tour="model-training"]', side: 'bottom', align: 'start' },
    { titleKey: 'tour.dataCleanup.title', descKey: 'tour.dataCleanup.desc', element: '[data-tour="data-cleanup"]', side: 'top', align: 'start' },
  ],
}

function buildSteps(page: string): DriveStep[] {
  const defs = TOUR_STEPS[page]
  if (!defs) return []
  const t = i18n.t.bind(i18n)
  return defs.map((d) => ({
    element: d.element,
    popover: {
      title: t(d.titleKey),
      description: t(d.descKey),
      side: d.side ?? 'bottom',
      align: d.align ?? 'start',
    },
  }))
}

interface TourContextValue {
  startTour: (page: string) => void
  isTourActive: boolean
}

const TourContext = createContext<TourContextValue>({
  startTour: () => {},
  isTourActive: false,
})

export function TourProvider({ children }: { children: ReactNode }) {
  const driverRef = useRef<ReturnType<typeof driver> | null>(null)
  const [isTourActive, setIsTourActive] = useState(false)
  const currentTourPage = useRef<string | null>(null)

  const startTour = useCallback((page: string) => {
    const defs = TOUR_STEPS[page]
    if (!defs) return

    currentTourPage.current = page

    if (driverRef.current) {
      driverRef.current.destroy()
    }

    const steps = buildSteps(page)

    const driverObj = driver({
      showProgress: true,
      animate: true,
      overlayColor: 'rgba(0, 0, 0, 0.5)',
      stagePadding: 8,
      stageRadius: 8,
      popoverClass: 'cafe-tour-popover',
      showButtons: ['next', 'previous', 'close'],
      disableActiveInteraction: true,
      onDestroyStarted: () => {
        if (!driverObj.hasNextStep() || driverObj.isLastStep()) {
          markCompleted(page)
          setIsTourActive(false)
          currentTourPage.current = null
        }
        driverObj.destroy()
      },
    } as Config)

    driverRef.current = driverObj
    driverObj.setSteps(steps)
    driverObj.drive()
    setIsTourActive(true)
  }, [])

  useEffect(() => {
    return () => {
      if (driverRef.current) {
        driverRef.current.destroy()
      }
    }
  }, [])

  const value = useMemo(() => ({ startTour, isTourActive }), [startTour, isTourActive])

  return (
    <TourContext value={value}>
      {children}
    </TourContext>
  )
}

export function useTour() {
  return useContext(TourContext)
}

export { resetAll, hasCompleted }
