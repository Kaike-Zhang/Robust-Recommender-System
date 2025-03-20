# Robust Recommender System: A Survey and Future Directions

Survey Link: [Link to the paper on Arxiv](https://arxiv.org/abs/2309.02057)

With the rapid growth of information, recommender systems have become integral for providing personalized suggestions and overcoming information overload. However, their practical deployment often encounters "dirty" data, where noise or malicious information can lead to abnormal recommendations. Research on improving recommender systems' robustness against such dirty data has thus gained significant attention. This survey provides a comprehensive review of recent work on recommender systems' robustness. We first present a taxonomy to organize current techniques for withstanding malicious attacks and natural noise. We then explore state-of-the-art methods in each category, including fraudster detection, adversarial training, certifiable robust training against malicious attacks, and regularization, purification, self-supervised learning against natural noise. Additionally, we summarize evaluation metrics and common datasets used to assess robustness. We discuss robustness across varying recommendation scenarios and its interplay with other properties like accuracy, interpretability, privacy, and fairness. Finally, we delve into open issues and future research directions in this emerging field. Our goal is to equip readers with a holistic understanding of robust recommender systems and spotlight pathways for future research and development.

## Citation
If you find our work useful, please cite our paper using the following BibTeX:

```bibtex
@article{zhang2023robust,
  title={Robust recommender system: a survey and future directions},
  author={Zhang, Kaike and Cao, Qi and Sun, Fei and Wu, Yunfan and Tao, Shuchang and Shen, Huawei and Cheng, Xueqi},
  journal={arXiv preprint arXiv:2309.02057},
  year={2023}
}
```


## Categories Index

- [Against Malicious Attacks](#against-malicious-attacks)
  - [Fraudster Detection](#fraudster-detection)
    - [Pre-Processing Detection](#pre-processing-detection)
    - [In-Processing Detection](#in-processing-detection)
    - [Post-Processing Detection](#post-processing-detection)
  - [Adversarial Training on](#adversarial-training-on)
    - [Parameters](#parameters)
    - [User Profile](#user-profile)
    - [Item Profile](#item-profile)
    - [Interaction](#interaction)
  - [Certifiable Robustness](#certifiable-robustness)
- [Against Natural Noise](#against-natural-noise)
  - [Regularization](#regularization)
  - [Purification](#purification)
  - [Self-supervised Learning](#self-supervised-learning)
    - [Generative-based](#generative-based)
    - [Contrastive-based](#contrastive-based)


If there are any robust recommender system methods that we may have missed, please feel free to contact us for additions.

# Against Malicious Attacks

## Fraudster Detection

### Pre-Processing Detection
| Title | Conference/Journal | Year | Link |
| --- | --- | --- | --- |
| Detecting shilling groups in online recommender systems based on graph convolutional network | IPM | 2022 | [Link](https://www.sciencedirect.com/science/article/pii/S0306457322001418) |
| Recommending Inferior Results: A General and Feature-Free Model for Spam Detection. | CIKM | 2020 | [Link](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.1100.0440) |
| Quick and accurate attack detection in recommender systems through user attributes. | RecSys | 2019 | [Link](https://dl.acm.org/doi/abs/10.1145/3298689.3347050) |
| Spotting anomalous ratings for rating systems by analyzing target users and items. | Neurocomputing | 2017 | [Link](https://www.sciencedirect.com/science/article/pii/S0925231217303430) |
| Re-scale AdaBoost for attack detection in collaborative filtering recommender systems. | Knowledge-Based Systems | 2016 | [Link](https://www.sciencedirect.com/science/article/pii/S0950705116000861) |
| SVM-TIA a shilling attack detection method based on SVM and target item analysis in recommender systems. | Neurocomputing | 2016 | [Link](https://www.sciencedirect.com/science/article/am/pii/S0925231216306038) |
| Catch the black sheep: unified framework for shilling attack detection based on fraudulent action propagation. | IJCAI | 2015 | [Link](https://www.ijcai.org/Proceedings/15/Papers/341.pdf) |
| Shilling attacks detection in recommender systems based on target item analysis.  | Plos | 2015 | [Link](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0130968&type=printable) |
| HHT–SVM: An online method for detecting profile injection attacks in collaborative recommender systems. | Knowledge-Based Systems | 2014 | [Link](https://www.sciencedirect.com/science/article/pii/S0950705114001427) |
| Detection of shilling attacks in recommender systems via spectral clustering.  | FUSION | 2014 | [Link](http://confcats_isif.s3.amazonaws.com/web-files/event/proceedings/html/2014Proceedings/papers/fusion2014_submission_78/paper78.pdf) |
| Detection of abnormal profiles on group attacks in recommender systems. | SIGIR | 2014 | [Link](https://dl.acm.org/doi/abs/10.1145/2600428.2609483) |
| Shilling attack detection utilizing semi-supervised learning method for collaborative recommender system.  | World Wide Web | 2013 | [Link](https://link.springer.com/article/10.1007/s11280-012-0164-6) |
| A belief propagation approach for detecting shilling attacks in collaborative filtering. | CIKM | 2013 | [Link](https://fekri.ece.gatech.edu/Publications/2013-11.pdf) |
| Shilling attack detection—a new approach for a trustworthy recommender system.  | INFORMS Journal on Computing | 2012 | [Link](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.1100.0440) |
| HySAD: A semi-supervised hybrid shilling attack detector for trustworthy product recommendation. | SIGKDD | 2012 | [Link](https://dl.acm.org/doi/abs/10.1145/2339530.2339684) |
| A Meta-learning-based Approach for Detecting Profile Injection Attacks in Collaborative Recommender Systems. | Jornal of Compuers | 2012 | [Link](http://www.jcomputers.us/vol7/jcp0701-26.pdf) |

### In-Processing Detection
| Title | Conference/Journal | Year | Link |
| --- | --- | --- | --- |
| Lorec: Combating poisons with large language model for robust sequential recommendation | SIGIR | 2024 | [Link](https://dl.acm.org/doi/pdf/10.1145/3626772.3657684) |
| Gcn-based user representation learning for unifying robust recommendation and fraudster detection. | SIGIR | 2020 | [Link](https://arxiv.org/pdf/2005.10150) |

### Post-Processing Detection
| Title | Conference/Journal | Year | Link |
| --- | --- | --- | --- |
| Adversarial attacks and detection on reinforcement learning-based interactive recommender systems | SIGIR | 2020 | [Link](https://arxiv.org/pdf/2006.07934) |

## Adversarial Training on

### Parameters
| Title | Conference/Journal | Year | Link |
| --- | --- | --- | --- |
| Understanding and Improving Adversarial Collaborative Filtering for Robust Recommendation | NeurIPS | 2025 | [Link](https://proceedings.neurips.cc/paper_files/paper/2024/file/da07cfa60cc883c5ee94ba899383bb6d-Paper-Conference.pdf) |
| Improving the Shortest Plank: Vulnerability-Aware Adversarial Training for Robust Recommender System | RecSys | 2024 | [Link](https://dl.acm.org/doi/pdf/10.1145/3640457.3688120) |
| Towards robust neural graph collaborative filtering via structure denoising and embedding perturbation. | TOIS | 2023 | [Link](https://dl.acm.org/doi/pdf/10.1145/3568396) |
| Adversarial Collaborative Filtering for Free | RecSys | 2023 | [Link](https://arxiv.org/pdf/2308.13541) |
| Adversarial learning to compare: Self-attentive prospective customer recommendation in location based social networks. | WSDM | 2020 | [Link](https://dl.acm.org/doi/pdf/10.1145/3336191.3371841) |
| Adversarial tensor factorization for context-aware recommendation.  | RecSys | 2019 | [Link](https://dl.acm.org/doi/abs/10.1145/3298689.3346987) |
| Adversarial mahalanobis distance-based attentive song recom-mender for automatic playlist continuation. | SIGIR | 2019 | [Link](https://dl.acm.org/doi/pdf/10.1145/3331184.3331234) |
| Adversarial collaborative auto-encoder for top-n recommendation.  | IJCNN | 2019 | [Link](https://arxiv.org/pdf/1808.05361) |
| Adversarial collaborative neural network for robust recommen-dation. | SIGIR | 2019 | [Link](https://dl.acm.org/doi/abs/10.1145/3331184.3331321) |
|  Enhancing the robustness of neural collaborative filtering systems under malicious attacks. | IEEE Transactions on Multimedia | 2018 | [Link](https://ieeexplore.ieee.org/abstract/document/8576563/) |
| Adversarial personalized ranking for recommenda-tion.  | SIGIR | 2018 | [Link](https://arxiv.org/pdf/1808.03908) |

### User Profile
| Title | Conference/Journal | Year | Link |
| --- | --- | --- | --- |
|  Fight fire with fire: towards robust recommender systems via adversarial poisoning training. | SIGIR | 2021 | [Link](https://dl.acm.org/doi/abs/10.1145/3404835.3462914) |

### Item Profile
| Title | Conference/Journal | Year | Link |
| --- | --- | --- | --- |
| A study of defensive methods to protect visual recommendation against adversarial manipulation of images. | SIGIR | 2021 | [Link](https://dl.acm.org/doi/abs/10.1145/3404835.3462848) |
| Adversarial training towards robust multimedia recommender system. | TKDE | 2019 | [Link](https://arxiv.org/pdf/1809.07062) |

### Interaction
| Title | Conference/Journal | Year | Link |
| --- | --- | --- | --- |
| Adversarial Graph Perturba-tions for Recommendations at Scale. | SIGIR | 2022 | [Link](https://dl.acm.org/doi/abs/10.1145/3477495.3531763) |
| Defending Substitution-Based Profile Pollution Attacks on Sequential Recommenders. | RecSys | 2022 | [Link](https://dl.acm.org/doi/abs/10.1145/3523227.3546770) |
| Exploring missing interactions: A convolutional generative adversarial network for collaborative filtering | CIKM | 2020 | [Link](https://dl.acm.org/doi/abs/10.1145/3340531.3411917) |

## Certifiable Robustness
| Title | Conference/Journal | Year | Link |
| --- | --- | --- | --- |
| Certifiable robustness to discrete adversarial perturbations for factorization machines | SIGIR | 2020 | [Link](https://dl.acm.org/doi/abs/10.1145/3397271.3401087) |

# Against Natural Noise

## Regularization
| Title | Conference/Journal | Year | Link |
| --- | --- | --- | --- |
| Denoising self-attentive sequential recommendation | RecSYs | 2022 | [Link](https://dl.acm.org/doi/abs/10.1145/3523227.3546788) |
| Robust collaborative filtering based on non-negative matrix factorization and R1-norm.  | KBS | 2017 | [Link](https://www.sciencedirect.com/science/article/pii/S0950705116304774) |

## Purification
| Title | Conference/Journal | Year | Link |
| --- | --- | --- | --- |
| Personalized Denoising Implicit Feedback for Robust Recommender System | WWW | 2025 | [Link](https://arxiv.org/pdf/2502.00348) |
| Neural Kalman Filtering for Robust Temporal Recommendation | WSDM | 2024 | [Link](https://dl.acm.org/doi/abs/10.1145/3616855.3635837) |
| Double Correction Framework for Denoising Recommendation | SIGKDD | 2024 | [Link](https://dl.acm.org/doi/abs/10.1145/3637528.3671692) |
| Towards robust neural graph collaborative filtering via structure denoising and embedding perturbation. | TOIS | 2023 | [Link](https://dl.acm.org/doi/pdf/10.1145/3568396) |
| Efficient Bi-Level Optimization for Recommendation Denoising | SIGKDD | 2023 | [Link](https://dl.acm.org/doi/abs/10.1145/3580305.3599324) |
| Robust Preference-Guided Denoising for Graph based Social Recommendation | WWW | 2023 | [Link](https://dl.acm.org/doi/abs/10.1145/3543507.3583374) |
| Denoising self-attentive sequential recommendation | RecSYs | 2022 | [Link](https://dl.acm.org/doi/abs/10.1145/3523227.3546788) |
| Self-guided learning to denoise for robust recommendation. | SIGIR | 2022 | [Link](https://arxiv.org/pdf/2204.06832) |
| Learning to Denoise Unreliable Interactions for Graph Collaborative Filtering.  | SIGIR | 2022 | [Link](https://dl.acm.org/doi/abs/10.1145/3477495.3531889) |
| Denoising Time Cycle Modeling for Recommendation | SIGIR | 2022 | [Link](https://arxiv.org/pdf/2402.02718) |
| Denoising user-aware memory network for recommendation | RecSys | 2021 | [Link](https://arxiv.org/pdf/2107.05474) |
| Concept-aware denoising graph neural network for micro-video recommendation. | CIKM | 2021 | [Link](https://arxiv.org/pdf/2109.13527) |
| The world is binary: Contrastive learning for denoising next basket recommendation.  | SIGIR | 2021 | [Link](https://dl.acm.org/doi/abs/10.1145/3404835.3462836) |
| Denoising implicit feedback for recommendation. | WSDM | 2021 | [Link](https://dl.acm.org/doi/pdf/10.1145/3437963.3441800) |
| Implicit Feedbacks are Not Always Favorable: Iterative Relabeled One-Class Collaborative Filtering against Noisy Interactions | MM | 2021 | [Link](https://dl.acm.org/doi/abs/10.1145/3474085.3475446) |
| A fuzzy model for managing natural noise in recommender systems. | Applied Soft Computing | 2016 | [Link](https://www.sciencedirect.com/science/article/pii/S1568494615007048) |
| Preference-based user rating correction process for interactive recommenda-tion systems. | Multimedia Tools and Applications | 2013 | [Link](https://link.springer.com/article/10.1007/s11042-012-1119-8) |

## Self-supervised Learning

### Generative-based
| Title | Conference/Journal | Year | Link |
| --- | --- | --- | --- |
| RecDiff: Diffusion Model for Social Recommendation | CIKM | 2024 | [Link](https://dl.acm.org/doi/abs/10.1145/3627673.3679630) |
| DiffKG: Knowledge Graph Diffusion Model for Recommendation | WSDM | 2024 | [Link](https://arxiv.org/pdf/2312.16890) |
| Multi-view denoising graph auto-encoders on heterogeneous information networks for cold-start recommendation | SIGKDD | 2021 | [Link](https://dl.acm.org/doi/abs/10.1145/3447548.3467427) |
| Recvae: A new variational autoencoder for top-n recommendations with implicit feedback. | WWW | 2020 | [Link](https://dl.acm.org/doi/abs/10.1145/3336191.3371831) |
| Collaborative denoising auto-encoders for top-n recommender systems. | WSDM | 2016 | [Link](https://dl.acm.org/doi/abs/10.1145/2835776.2835837) |
| Deep collaborative filtering via marginalized denoising auto-encoder | CIKM | 2015 | [Link](https://dl.acm.org/doi/abs/10.1145/2806416.2806527) |

### Contrastive-based
| Title | Conference/Journal | Year | Link |
| --- | --- | --- | --- |
| Self-Supervised Denoising through Independent Cascade Graph Augmentation for Robust Social Recommendation | SIGKDD | 2024 | [Link](https://dl.acm.org/doi/abs/10.1145/3637528.3671958) |
| Adaptive Graph Contrastive Learning for Recommendation | SIGKDD | 2023 | [Link](https://arxiv.org/pdf/2305.10837) |
| Knowledge-Adaptive Contrastive Learning for Recommendation | WSDM | 2023 | [Link](https://dl.acm.org/doi/abs/10.1145/3539597.3570483) |
| Denoised Self-Augmented Learning for Social Recommendation. | ArXiv | 2023 | [Link](https://arxiv.org/pdf/2305.12685) |
| Intent contrastive learning for sequential recommendation | WWW | 2022 | [Link](https://arxiv.org/pdf/2202.02519) |
|  Learning to Denoise Unreliable Interactions for Graph Collaborative Filtering. | SIGIR | 2022 | [Link](https://dl.acm.org/doi/abs/10.1145/3477495.3531889) |
|  Learning Robust Recommenders through Cross-Model Agreement. | WWW | 2022 | [Link](https://arxiv.org/pdf/2105.09605) |
| Knowledge graph contrastive learning for recommendation | SIGIR | 2022 | [Link](https://arxiv.org/pdf/2205.00976) |
| Contrastive self-supervised sequential recommendation with robust augmentation. | Arxiv | 2021 | [Link](https://arxiv.org/pdf/2108.06479) |
| Self-supervised graph learning for recommendation | SIGIR | 2021 | [Link](https://arxiv.org/pdf/2010.10783) |
|  S3-rec: Self-supervised learning for sequential recommendation with mutual information maximization.  | CIKM | 2020 | [Link](https://dl.acm.org/doi/abs/10.1145/3340531.3411954) |
| Collaborative filtering with stacked denoising autoencoders and sparse inputs. | NeurIPS workshop | 2015 | [Link](https://inria.hal.science/hal-01256422/) |

