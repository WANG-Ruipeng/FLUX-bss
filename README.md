# FLUX Fill BSS/BDS Condition-Anchoring Experiment

This repository is a lightweight experiment scaffold for a FLUX.1 Fill-dev
condition-anchoring probe. It stores code, manifests, schemas, notebooks,
reports, and small reproducible metadata. Model weights, generated images,
metrics outputs, figures, and other heavy artifacts should live in Google
Drive or a local ignored run folder.

Primary model:

```text
black-forest-labs/FLUX.1-Fill-dev
```

Default Drive locations used by the Colab notebook:

```text
/content/drive/MyDrive/Colab_Projects/FLUX-bss/models/FLUX.1-Fill-dev/
/content/drive/MyDrive/Colab_Projects/FLUX-bss/runs/flux_fill_bss_bds_v1/
```

The notebook asks for a Hugging Face token only when the Drive weight folder is
missing or an explicit re-download is requested. Tokens are not written to code
or reports.

Local dry-run checks do not download weights or run FLUX inference.
