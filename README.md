# Signify Your AI: American Sign Language & Machine Learning

A convolutional neural network (CNN) research project exploring how different image preprocessing filters affect the accuracy and generalizability of American Sign Language (ASL) alphabet recognition.

Built by Isaiah Cerven, Alex Greene, and William Beckhorn.

---

## Research question

> *How will changing filters in a CNN improve the generalizability of CNNs for ASL recognition?*

Existing ASL recognition models exist, but improving their accuracy and ability to generalize across different conditions has not been thoroughly explored. This project investigates whether different preprocessing approaches to input images meaningfully change model performance.

---

## Approach

A standardized CNN was trained and evaluated across four different preprocessing configurations, holding all other variables constant.

**Dataset:** 8,700-image subset of a 67,000-image ASL alphabet dataset, split into train, validation, test, and real-life test sets.

**Preprocessing applied to all configurations:**
- `transforms.ToTensor()`
- `transforms.Normalize()`

**Filters tested:**

| Filter | Description | Input channels |
|---|---|---|
| RGB | Raw color image | 3 |
| Grayscale | Color removed | 1 |
| Hand Landmark | MediaPipe-detected hand landmarks drawn on a black image — no original photo used | — |
| Transformations | Rotation, scaling, flipping, cropping augmentations | — |

---

## Results

Evaluated on the validation set across 5 epochs:

| Filter | Training accuracy | Test loss | Test accuracy |
|---|---|---|---|
| RGB | 83.76% | 0.3584 | **88%** |
| Grayscale | 79.03% | 0.4613 | 83% |
| Hand Landmark | 69.43% | 1.4755 | 60% |

---

## Tech stack

| Layer | Tools |
|---|---|
| Language | Python, Jupyter Notebook |
| ML framework | PyTorch (`ImageFolder`, `DataLoader`) |
| Hand detection | MediaPipe |
| Team size | 3 |

---

## Known limitations

- **Letters J and Z** are non-static in ASL and require motion to distinguish — the model only processes still images, making reliable classification of these letters impossible with this approach
- **MediaPipe** occasionally fails to detect hand landmark coordinates, causing the hand landmark filter to fall back or error
- **DataLoader shuffle** must be set to `True` — without it, the model trains on letters in alphabetical folder order and learns to predict the final letter for all inputs
