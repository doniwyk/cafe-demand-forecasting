import { useEffect } from 'react'
import { useRouterState } from '@tanstack/react-router'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { useTour, hasCompleted } from '@/contexts/tour-context'
import { CircleHelpIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

export function AppTour() {
  const pathname = useRouterState({ select: (s) => s.location.pathname })
  const { startTour, isTourActive } = useTour()
  const { t } = useTranslation()

  useEffect(() => {
    if (isTourActive) return
    if (!hasCompleted(pathname)) {
      const timer = setTimeout(() => {
        startTour(pathname)
      }, 800)
      return () => clearTimeout(timer)
    }
  }, [pathname, isTourActive, startTour])

  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <button
            type="button"
            className="inline-flex shrink-0 items-center justify-center size-8 rounded-lg text-muted-foreground hover:bg-accent hover:text-accent-foreground disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg]:size-4"
            onClick={() => startTour(pathname)}
            disabled={isTourActive}
          >
            <CircleHelpIcon className="size-4" />
          </button>
        }
      />
      <TooltipContent>
        <p>{t('tour.startTour')}</p>
      </TooltipContent>
    </Tooltip>
  )
}
