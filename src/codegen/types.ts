import type { LibraryId } from '../state/types';

export interface CodeBundle {
  library: LibraryId;
  title: string;
  language: 'python';
  source: string;
}

export interface CodegenOptions {
  /** Include dataset generation helpers in the source. */
  withData?: boolean;
  /** Indent size for generated Python. */
  indent?: string;
}
