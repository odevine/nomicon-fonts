// Column sorting for tables marked data-sortable. Cells carry their sort key in data-v, an empty key sorts last

const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });

for (const table of document.querySelectorAll<HTMLTableElement>('table[data-sortable]')) {
  const body = table.tBodies[0];
  const heads = [...table.tHead!.rows[0].cells];
  // the export order, restored when a column's third press clears its sort
  const original = [...body.rows];
  heads.forEach((th, col) => {
    const text = th.dataset.type === 'text';
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'sort';
    button.append(...th.childNodes);
    th.append(button);
    button.addEventListener('click', () => {
      // text sorts A to Z first, numbers largest first
      const first = text ? 'ascending' : 'descending', second = text ? 'descending' : 'ascending';
      const now = th.getAttribute('aria-sort');
      const next = now === first ? second : now === second ? null : first;
      for (const h of heads) h.removeAttribute('aria-sort');
      if (!next) { body.append(...original); return; }
      th.setAttribute('aria-sort', next);
      const key = (r: HTMLTableRowElement) => r.cells[col].dataset.v ?? r.cells[col].textContent ?? '';
      const sign = next === 'ascending' ? 1 : -1;
      const rows = [...original].sort((a, b) => {
        const x = key(a), y = key(b);
        if (x === '' || y === '') return (x === '' ? 1 : 0) - (y === '' ? 1 : 0);
        return sign * (text ? collator.compare(x, y) : Number(x) - Number(y));
      });
      body.append(...rows);
    });
  });
}
