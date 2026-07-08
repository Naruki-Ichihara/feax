import React, {useEffect, useRef, useState} from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import useBaseUrl from '@docusaurus/useBaseUrl';
import {useColorMode} from '@docusaurus/theme-common';

/**
 * Embeds the self-contained interactive solver benchmark (static/benchmarks/dashboard.html)
 * as a theme-synced, auto-resizing iframe. The dashboard reports its content height and
 * accepts light/dark from the surrounding Docusaurus color mode via postMessage.
 */
function Frame({minHeight}: {minHeight: number}) {
  const {colorMode} = useColorMode();
  const src = useBaseUrl('/benchmarks/dashboard.html');
  const ref = useRef<HTMLIFrameElement>(null);
  const [height, setHeight] = useState(minHeight);

  // auto-height: dashboard posts its CONTENT height {feaxHeight}. Only react to
  // changes > 1px so sub-pixel rounding can't drive a grow-forever feedback loop.
  useEffect(() => {
    const onMsg = (e: MessageEvent) => {
      const h = e?.data?.feaxHeight;
      if (typeof h === 'number' && h > 0) {
        const target = Math.max(minHeight, Math.ceil(h));
        setHeight((prev) => (Math.abs(target - prev) > 1 ? target : prev));
      }
    };
    window.addEventListener('message', onMsg);
    return () => window.removeEventListener('message', onMsg);
  }, [minHeight]);

  // theme sync: push colorMode into the iframe
  useEffect(() => {
    ref.current?.contentWindow?.postMessage({feaxTheme: colorMode}, '*');
  }, [colorMode]);

  return (
    <iframe
      ref={ref}
      src={`${src}?theme=${colorMode}`}
      title="FEAX GPU solver benchmark"
      loading="lazy"
      scrolling="no"
      onLoad={() => ref.current?.contentWindow?.postMessage({feaxTheme: colorMode}, '*')}
      style={{
        display: 'block',
        width: '100%',
        height,
        border: '1px solid var(--ifm-color-emphasis-200)',
        borderRadius: 12,
        overflow: 'hidden',
        colorScheme: 'auto',
      }}
    />
  );
}

export default function BenchmarkEmbed({minHeight = 640}: {minHeight?: number}) {
  return (
    <BrowserOnly fallback={<div style={{padding: 40, textAlign: 'center'}}>Loading benchmark…</div>}>
      {() => <Frame minHeight={minHeight} />}
    </BrowserOnly>
  );
}
