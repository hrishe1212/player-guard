import os
import re
import math
import zipfile
import xml.etree.ElementTree as ET

class PolicyRAG:
    def __init__(self, docx_path='PlayerGuard_AI_Policy.docx'):
        self.docx_path = docx_path
        self.chunks = []
        self._load_and_index()

    def _extract_text_from_docx(self):
        if not os.path.exists(self.docx_path):
            return []
        try:
            with zipfile.ZipFile(self.docx_path) as z:
                xml_content = z.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            paragraphs = []
            for p in tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                p_text = ''.join(node.text for node in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text)
                if p_text.strip():
                    paragraphs.append(p_text.strip())
            return paragraphs
        except Exception as e:
            print(f"Error reading docx: {e}")
            return []

    def _load_and_index(self):
        paragraphs = self._extract_text_from_docx()
        chunks = []
        current_rule = None
        current_lines = []

        for p in paragraphs:
            m = re.match(r'^([A-Z]{2,4}-\d{3})\s*[—\-–]\s*(.+)', p)
            if m:
                if current_rule:
                    chunks.append({
                        'rule_id': current_rule[0],
                        'title': current_rule[1],
                        'content': '\n'.join(current_lines),
                        'tokens': set(re.findall(r'\w+', '\n'.join(current_lines).lower()))
                    })
                current_rule = (m.group(1), m.group(2))
                current_lines = [p]
            elif current_rule:
                current_lines.append(p)

        if current_rule:
            chunks.append({
                'rule_id': current_rule[0],
                'title': current_rule[1],
                'content': '\n'.join(current_lines),
                'tokens': set(re.findall(r'\w+', '\n'.join(current_lines).lower()))
            })

        self.chunks = chunks

    def retrieve(self, query, risk_label=None, symptoms="", top_k=4):
        """
        Retrieves relevant policy rules based on query, risk prediction, and symptoms.
        Guarantees inclusion of PW-001 (Human Authority) and relevant ESC/MOD rules.
        """
        if not self.chunks:
            return []

        search_tokens = set(re.findall(r'\w+', (query + " " + symptoms).lower()))
        if risk_label == 1:
            search_tokens.update(['elevated', 'doctor', 'escalation', 'urgent', 'restriction'])
        elif risk_label == 0:
            search_tokens.update(['lower', 'monitoring', 'routine', 'observation'])

        scored_chunks = []
        for c in self.chunks:
            # Token overlap score
            overlap = len(search_tokens.intersection(c['tokens']))
            score = overlap / (math.log(len(c['tokens']) + 1) + 1)

            # Boost mandatory governance rules
            if c['rule_id'] == 'PW-001':
                score += 3.0  # Always high priority
            if risk_label == 1 and c['rule_id'] in ['ESC-001', 'MOD-003', 'AUD-001']:
                score += 2.5
            elif risk_label == 0 and c['rule_id'] in ['MOD-004', 'SYM-001', 'MON-001']:
                score += 2.0
            if symptoms and c['rule_id'] in ['ESC-002', 'SYM-001', 'SYM-002']:
                score += 3.0

            scored_chunks.append((score, c))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        results = [item[1] for item in scored_chunks[:top_k]]
        
        # Ensure PW-001 is included
        rule_ids = [r['rule_id'] for r in results]
        if 'PW-001' not in rule_ids:
            pw001 = next((c for c in self.chunks if c['rule_id'] == 'PW-001'), None)
            if pw001:
                results.insert(0, pw001)
                results = results[:top_k]

        return results

if __name__ == '__main__':
    rag = PolicyRAG()
    print(f"RAG Initialized with {len(rag.chunks)} policy rules.")
    results = rag.retrieve("player fatigue elevated heart rate", risk_label=1, symptoms="sore hamstring")
    print(f"\nTop Retrieved Rules:")
    for r in results:
        print(f"- [{r['rule_id']}] {r['title']}")
