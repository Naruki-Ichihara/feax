import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

export default function HeroSection(): ReactNode {
  return (
    <div className={styles.hero}>
      <img
        src={useBaseUrl('/img/logo.svg')}
        alt="FEAX Logo"
        className={styles.heroLogo}
      />
      <Heading as="h1" className={styles.heroTitle}>
        FEAX
      </Heading>
      <p className={styles.heroSubtitle}>
        A fully differentiable finite element engine built on{' '}
        <a href="https://github.com/google/jax" target="_blank" rel="noopener noreferrer">
          JAX
        </a>
        , designed for gradient-based optimization and machine learning
        on PDE simulations.
      </p>
      <div className={styles.buttons}>
        <Link
          className={clsx('button button--lg', styles.buttonPrimary)}
          to="/getting-started">
          Get Started
        </Link>
        <Link
          className={clsx('button button--lg', styles.buttonSecondary)}
          to="/getting-started/installation">
          Installation
        </Link>
      </div>
    </div>
  );
}

type FeatureItem = {
  title: string;
  icon: string;
  description: ReactNode;
};

const features: FeatureItem[] = [
  {
    title: 'JAX Transformations',
    icon: '\u2699\uFE0F',
    description: (
      <>
        All solvers work seamlessly with <code>jax.jit</code>,{' '}
        <code>jax.grad</code>, and <code>jax.vmap</code>, and arbitrary
        compositions such as <code>jit(grad(...))</code>.
      </>
    ),
  },
  {
    title: 'GPU Direct Solver',
    icon: '\uD83D\uDE80',
    description: (
      <>
        Native cuDSS integration for sparse direct solves on GPU, with
        automatic matrix property detection (General / Symmetric / SPD).
      </>
    ),
  },
  {
    title: 'End-to-End Differentiability',
    icon: '\u2202',
    description: (
      <>
        Gradients flow through assembly, boundary conditions, linear/nonlinear
        solvers, and post-processing &mdash; enabling topology optimization,
        inverse problems, and physics-informed learning.
      </>
    ),
  },
];

export function FeatureCards(): ReactNode {
  return (
    <div className={styles.features}>
      {features.map((item, idx) => (
        <div key={idx} className={styles.featureCard}>
          <div className={styles.featureIcon}>{item.icon}</div>
          <Heading as="h3" className={styles.featureTitle}>{item.title}</Heading>
          <p className={styles.featureDescription}>{item.description}</p>
        </div>
      ))}
    </div>
  );
}
