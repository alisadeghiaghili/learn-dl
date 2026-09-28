import type { ExperimentState, LibraryId } from '../state/types';
import { generateKeras } from './keras';
import { generatePyTorch } from './pytorch';
import { generateTensorFlow } from './tensorflow';
import type { CodeBundle, CodegenOptions } from './types';

export type { CodeBundle, CodegenOptions } from './types';
export { generateKeras } from './keras';
export { generatePyTorch } from './pytorch';
export { generateTensorFlow } from './tensorflow';

/**
 * Generate code for one library.
 */
export function generateCode(
  library: LibraryId,
  state: ExperimentState,
  options?: CodegenOptions,
): CodeBundle {
  switch (library) {
    case 'pytorch':
      return generatePyTorch(state, options);
    case 'tensorflow':
      return generateTensorFlow(state, options);
    case 'keras':
      return generateKeras(state, options);
  }
}

/**
 * Generate all three libraries from one experiment state.
 */
export function generateAllCode(
  state: ExperimentState,
  options?: CodegenOptions,
): Record<LibraryId, CodeBundle> {
  return {
    pytorch: generatePyTorch(state, options),
    tensorflow: generateTensorFlow(state, options),
    keras: generateKeras(state, options),
  };
}
