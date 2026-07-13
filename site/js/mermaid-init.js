// Initialize Mermaid with default config
mermaid.initialize({
  startOnLoad: true,
  theme: 'default',
  securityLevel: 'loose',
});

// Re-render Mermaid diagrams on instant navigation
// Material for MkDocs exposes a 'document$' observable
if (window.document$) {
  window.document$.subscribe(function() {
    mermaid.contentLoaded();
  });
}
