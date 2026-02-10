# Tasks

Tasks define **what** you want to extract. Models define **how**.

---

## Available Tasks

| Task | Input | Output | Status |
|------|-------|--------|--------|
| [Text Extraction](text-extraction.md) | Image / PDF | Markdown, HTML | ✅ Ready |
| [Layout Analysis](layout-analysis.md) | Image | Bounding boxes + labels | ✅ Ready |
| [OCR](ocr.md) | Image | Text + coordinates | ✅ Ready |
| [Table Extraction](table-extraction.md) | Table image | Structured table data | ✅ Ready |
| [Reading Order](reading-order.md) | Layout + OCR | Ordered elements | ✅ Ready |
| [Structured Extraction](structured-extraction.md) | Image + Schema | Typed Pydantic objects | ✅ Ready |

---

## Choosing a Task

**"I want readable text from a PDF"**
→ [Text Extraction](text-extraction.md)

**"I need to know where tables and figures are"**
→ [Layout Analysis](layout-analysis.md)

**"I need word positions for downstream processing"**
→ [OCR](ocr.md)

**"I want structured data from a table"**
→ [Table Extraction](table-extraction.md)

**"I need elements in reading order"**
→ [Reading Order](reading-order.md)

**"I want typed data from invoices/forms"**
→ [Structured Extraction](structured-extraction.md)

---

## Upcoming Tasks

| Task | Description | Status |
|------|-------------|--------|
| **Math Recognition** | LaTeX from equations | 🔜 Soon |
| **Chart Understanding** | Data extraction from charts | 🔜 Planned |
| **Image Captioning** | Caption figures and images | 🔜 Planned |

See [Roadmap](../../ROADMAP.md) for full tracking.
