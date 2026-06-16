// Copy Page Content Functionality for MkDocs Material
(function() {
    'use strict';
  
    // Wait for DOM to be ready
    function initCopyButtons() {
      // Find the main content area
      const contentArea = document.querySelector('.md-content__inner') || 
                         document.querySelector('.md-content') ||
                         document.querySelector('article');
      
      if (!contentArea) {
        console.warn('Copy buttons: Content area not found');
        return;
      }
  
      // Find the page title area to insert buttons
      const titleElement = document.querySelector('.md-content__inner h1') ||
                          document.querySelector('h1');
      
      if (!titleElement) {
        console.warn('Copy buttons: Title not found');
        return;
      }
  
      // Create button container
      const buttonContainer = document.createElement('div');
      buttonContainer.className = 'copy-page-buttons';
      buttonContainer.style.cssText = `
        display: inline-flex;
        gap: 0.5rem;
        margin-left: 1rem;
        vertical-align: middle;
      `;
  
      // Create "Copy as Text" button
      const copyTextBtn = createCopyButton('Copy as Text', 'text');
      // Create "Copy as Markdown" button
      const copyMarkdownBtn = createCopyButton('Copy as Markdown', 'markdown');
  
      buttonContainer.appendChild(copyTextBtn);
      buttonContainer.appendChild(copyMarkdownBtn);
  
      // Insert buttons after the title
      titleElement.style.display = 'inline-block';
      titleElement.parentNode.insertBefore(buttonContainer, titleElement.nextSibling);
  
      // Function to create a copy button
      function createCopyButton(label, type) {
        const button = document.createElement('button');
        button.className = 'md-button md-button--primary';
        button.style.cssText = `
          font-size: 0.8rem;
          padding: 0.4rem 0.8rem;
          margin: 0;
          cursor: pointer;
          border-radius: 0.2rem;
        `;
        button.textContent = label;
        button.setAttribute('title', `Copy page content as ${type}`);
        
        button.addEventListener('click', async function(e) {
          e.preventDefault();
          e.stopPropagation();
          
          try {
            const content = type === 'markdown' 
              ? extractAsMarkdown(contentArea)
              : extractAsText(contentArea);
            
            await navigator.clipboard.writeText(content);
            
            // Show feedback
            const originalText = button.textContent;
            button.textContent = '✓ Copied!';
            button.style.backgroundColor = '#4caf50';
            
            setTimeout(() => {
              button.textContent = originalText;
              button.style.backgroundColor = '';
            }, 2000);
          } catch (err) {
            console.error('Failed to copy:', err);
            button.textContent = '✗ Failed';
            setTimeout(() => {
              button.textContent = label;
            }, 2000);
          }
        });
        
        return button;
      }
  
      // Extract content as plain text
      function extractAsText(element) {
        // Clone to avoid modifying the original
        const clone = element.cloneNode(true);
        
        // Remove unwanted elements
        const selectorsToRemove = [
          '.copy-page-buttons',
          '.md-header',
          '.md-sidebar',
          'script',
          'style',
          '.md-footer',
          '.md-nav',
          'nav'
        ];
        
        selectorsToRemove.forEach(selector => {
          clone.querySelectorAll(selector).forEach(el => el.remove());
        });
        
        // Get text content
        return clone.textContent || clone.innerText || '';
      }
  
      // Extract content as markdown (simplified conversion)
      function extractAsMarkdown(element) {
        const clone = element.cloneNode(true);
        
        // Remove unwanted elements
        const selectorsToRemove = [
          '.copy-page-buttons',
          '.md-header',
          '.md-sidebar',
          'script',
          'style',
          '.md-footer',
          'nav'
        ];
        
        selectorsToRemove.forEach(selector => {
          clone.querySelectorAll(selector).forEach(el => el.remove());
        });
        
        // Convert to markdown-like format
        let markdown = '';
        
        // Get the page title
        const h1 = clone.querySelector('h1');
        if (h1) {
          markdown += `# ${h1.textContent.trim()}\n\n`;
          h1.remove();
        }
        
        // Process all elements
        function processElement(el, indent = '') {
          if (!el) return '';
          
          let result = '';
          const tagName = el.tagName ? el.tagName.toLowerCase() : '';
          const text = el.textContent ? el.textContent.trim() : '';
          
          switch(tagName) {
            case 'h1':
              result += `\n# ${text}\n\n`;
              break;
            case 'h2':
              result += `\n## ${text}\n\n`;
              break;
            case 'h3':
              result += `\n### ${text}\n\n`;
              break;
            case 'h4':
              result += `\n#### ${text}\n\n`;
              break;
            case 'h5':
              result += `\n##### ${text}\n\n`;
              break;
            case 'h6':
              result += `\n###### ${text}\n\n`;
              break;
            case 'p':
              if (text) {
                // Check for code blocks inside
                const code = el.querySelector('code');
                if (code && !code.closest('pre')) {
                  result += text.replace(code.textContent, `\`${code.textContent}\``) + '\n\n';
                } else {
                  result += text + '\n\n';
                }
              }
              break;
            case 'pre':
              const codeEl = el.querySelector('code');
              if (codeEl) {
                const lang = codeEl.className.match(/language-(\w+)/) 
                  ? codeEl.className.match(/language-(\w+)/)[1] 
                  : '';
                result += `\n\`\`\`${lang}\n${codeEl.textContent}\n\`\`\`\n\n`;
              }
              break;
            case 'ul':
            case 'ol':
              el.querySelectorAll(':scope > li').forEach((li, idx) => {
                const prefix = tagName === 'ul' ? '- ' : `${idx + 1}. `;
                result += indent + prefix + li.textContent.trim() + '\n';
              });
              result += '\n';
              break;
            case 'blockquote':
              const lines = text.split('\n');
              lines.forEach(line => {
                if (line.trim()) {
                  result += `> ${line.trim()}\n`;
                }
              });
              result += '\n';
              break;
            case 'strong':
            case 'b':
              result += `**${text}**`;
              break;
            case 'em':
            case 'i':
              result += `*${text}*`;
              break;
            case 'code':
              if (!el.closest('pre')) {
                result += `\`${text}\``;
              }
              break;
            case 'a':
              const href = el.getAttribute('href') || '';
              result += `[${text}](${href})`;
              break;
            default:
              // For other elements, process children
              if (el.children.length === 0 && text) {
                result += text;
              } else {
                Array.from(el.childNodes).forEach(child => {
                  if (child.nodeType === Node.TEXT_NODE) {
                    const txt = child.textContent.trim();
                    if (txt) result += txt + ' ';
                  } else if (child.nodeType === Node.ELEMENT_NODE) {
                    result += processElement(child, indent);
                  }
                });
              }
          }
          
          return result;
        }
        
        // Process all child nodes
        Array.from(clone.childNodes).forEach(child => {
          if (child.nodeType === Node.ELEMENT_NODE) {
            markdown += processElement(child);
          } else if (child.nodeType === Node.TEXT_NODE) {
            const txt = child.textContent.trim();
            if (txt) markdown += txt + '\n\n';
          }
        });
        
        // Clean up extra newlines
        markdown = markdown.replace(/\n{3,}/g, '\n\n').trim();
        
        return markdown;
      }
    }
  
    // Initialize on page load
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', initCopyButtons);
    } else {
      initCopyButtons();
    }
  
    // Re-initialize on navigation (Material for MkDocs instant loading)
    if (window.document$) {
      window.document$.subscribe(function() {
        setTimeout(initCopyButtons, 100);
      });
    }
  })();