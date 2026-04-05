import { createContext, useContext, useState, type ReactNode } from 'react'

export const MODEL_TYPES = ['xgboost', 'random_forest', 'sarimax', 'prophet'] as const
export type ModelType = (typeof MODEL_TYPES)[number]

export const MODEL_LABELS: Record<ModelType, string> = {
  xgboost: 'XGBoost',
  random_forest: 'Random Forest',
  sarimax: 'SARIMAX',
  prophet: 'Prophet',
}

interface ModelContextValue {
  modelType: ModelType
  setModelType: (type: ModelType) => void
}

const ModelContext = createContext<ModelContextValue>({
  modelType: 'xgboost',
  setModelType: () => {},
})

export function ModelProvider({ children }: { children: ReactNode }) {
  const [modelType, setModelType] = useState<ModelType>('xgboost')
  return (
    <ModelContext value={{ modelType, setModelType }}>
      {children}
    </ModelContext>
  )
}

export function useModelType() {
  return useContext(ModelContext)
}
