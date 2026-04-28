## 1. Implementation steps

**Clone the repo**
```bash
git clone 
```

**Build the image**

```bash
docker build -t tsunami .
```

**Run a single interactive query**

```bash
docker run -it --rm tsunami python query_cli.py --dataset california_real
```

Once the index is built, type a query at the prompt:

```
SELECT COUNT(*) FROM data WHERE MedInc >= 7.0 AND HouseAge <= 20
```
