import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface AnswerBodyProps {
  text: string;
  /** Adds a caret while tokens are still arriving. */
  streaming?: boolean;
}

/**
 * The answer itself, rendered as markdown. Set in a serif at a generous
 * measure: the source material is investigation reports, and answers regularly
 * run to several paragraphs of technical prose that a chat-bubble sans at 14px
 * makes genuinely tiring to read.
 */
export function AnswerBody({ text, streaming = false }: AnswerBodyProps) {
  return (
    <div className={`answer-prose ${streaming ? 'streaming-caret' : ''}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // Tables appear in comparative answers and can exceed the measure;
          // let them scroll rather than forcing the page to.
          table: ({ children }) => (
            <div className="overflow-x-auto">
              <table>{children}</table>
            </div>
          ),
          a: ({ children, href }) => (
            <a href={href} target="_blank" rel="noreferrer noopener">
              {children}
            </a>
          ),
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
