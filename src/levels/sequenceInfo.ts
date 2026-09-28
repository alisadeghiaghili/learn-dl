import type { SequenceInfo } from './types';

export const sequenceInfo: SequenceInfo[] = [
  {
    id: 'foundations',
    tab: 'main',
    displayName: {
      en_US: 'Foundations',
      fa: 'مبانی',
    },
    about: {
      en_US: 'Tensors, shapes, autograd, and the first neuron',
      fa: 'تنسورها، شکل‌ها، autograd و اولین نورون',
    },
  },
  {
    id: 'mlp',
    tab: 'main',
    displayName: {
      en_US: 'Multilayer Perceptron',
      fa: 'پرسپترون چندلایه',
    },
    about: {
      en_US: 'Dense stacks, activations, losses, optimizers, the training loop',
      fa: 'لایه‌های dense، فعال‌سازی‌ها، loss، بهینه‌سازها، حلقه‌ی ترین',
    },
  },
  {
    id: 'regularization',
    tab: 'main',
    displayName: {
      en_US: 'Regularization',
      fa: 'منظم‌سازی',
    },
    about: {
      en_US: 'Dropout, batch norm, weight decay, early stopping',
      fa: 'دروپ‌اوت، نرمال‌سازی دسته‌ای، weight decay، توقف زودهنگام',
    },
  },
  {
    id: 'cnn',
    tab: 'main',
    displayName: {
      en_US: 'Convolutional Networks',
      fa: 'شبکه‌های کانولوشنی',
    },
    about: {
      en_US: 'Conv, pooling, spatial hierarchies',
      fa: 'کانولوشن، pooling، سلسله‌مراتب فضایی',
    },
  },
  {
    id: 'sequence',
    tab: 'main',
    displayName: {
      en_US: 'Sequence Models',
      fa: 'مدل‌های دنباله‌ای',
    },
    about: {
      en_US: 'RNN, LSTM, GRU and sequence tasks',
      fa: 'RNN، LSTM، GRU و کارهای دنباله‌ای',
    },
  },
  {
    id: 'attention',
    tab: 'main',
    displayName: {
      en_US: 'Attention & Transformers',
      fa: 'توجه و ترنسفورمرها',
    },
    about: {
      en_US: 'Attention, multi-head, transformer blocks',
      fa: 'توجه، چندسر، بلوک ترنسفورمر',
    },
  },
  {
    id: 'training',
    tab: 'main',
    displayName: {
      en_US: 'Training Craft',
      fa: 'فن ترین',
    },
    about: {
      en_US: 'LR schedules, init, clipping, mixed precision ideas',
      fa: 'زمانه‌بندی یادگیری، مقداردهی، clipping، ایده‌های mixed precision',
    },
  },
  {
    id: 'production',
    tab: 'main',
    displayName: {
      en_US: 'Transfer & Production',
      fa: 'انتقال و تولید',
    },
    about: {
      en_US: 'Transfer learning, fine-tuning, export and inference',
      fa: 'یادگیری انتقالی، fine-tune، export و استنتاج',
    },
  },
  {
    id: 'pytorch',
    tab: 'libraries',
    displayName: {
      en_US: 'PyTorch Track',
      fa: 'مسیر PyTorch',
    },
    about: {
      en_US: 'Module, autograd, DataLoader, custom loops',
      fa: 'Module، autograd، DataLoader، حلقه‌های سفارشی',
    },
  },
  {
    id: 'tensorflow',
    tab: 'libraries',
    displayName: {
      en_US: 'TensorFlow Track',
      fa: 'مسیر TensorFlow',
    },
    about: {
      en_US: 'tf.function, GradientTape, tf.data',
      fa: 'tf.function، GradientTape، tf.data',
    },
  },
  {
    id: 'keras',
    tab: 'libraries',
    displayName: {
      en_US: 'Keras Track',
      fa: 'مسیر Keras',
    },
    about: {
      en_US: 'Sequential/Functional, callbacks, preprocessing',
      fa: 'Sequential/Functional، callbackها، پیش‌پردازش',
    },
  },
];

export function getSequence(id: string): SequenceInfo | undefined {
  return sequenceInfo.find((s) => s.id === id);
}
