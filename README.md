# Encode Sentences into Gaussian Mixture for Unsupervised Sentence Similarity
* あるいは： Variational AutoEncoder meets Gaussian Mixture

## 目的
* 文の **類似性** を捉える，完全教師なし表現学習の方法を考える
* つまり，以下の条件を満たす sentence encoder を構築したい
	* 生コーパスのみを用いて学習可能
	* 文ペアの同義性，意味の類似性，内容の類似性を評価可能

## 背景
* 文間類似性の評価は，情報検索・文書分類・文間関係識別などの基礎をなす技術である
* 教師あり学習を用いる場合は，アノテーションのコストがかかる
	* 言語やドメインに応じてコーパスを作る必要がある(ex. Semantic Textual Similarity)
* 教師なし学習も数多く提案されている．特にWord Mover's Distanceに基づく手法[Kusner et al., 2015]は有力だが，語順や文脈の考慮ができない，単語重み付けの方法がheuristicである，といった課題を抱えている
* 最適輸送理論(=OMT)の進展により，確率分布間の距離を目的関数に含めることで，性能を改善できる事例が報告されている(WGAN[ARJOVSKY et al., 2017]，WAE[TOLSTIKHIN et al., 2018])．確率分布の距離計量という発想を，文間類似性タスクに持ち込む

## 新規性
* 提案手法は，Word Mover's Distance：最適輸送理論(=OMT)に基づく手法，の改善版である
* 具体的には，Stacked BiLSTM Encodrが，tokenごとにgaussian distributionを出力する．従って，文はgaussian mixtureにEncodeされる
* 文間の距離は，gaussian mixtureペアの距離として与える．これをapproximated wasserstein distanceで定量化する
* WMDに基づく手法に対する長所は，以下の通り
	* 語順を考慮できる
	* 文脈を考慮して，単語の表現および重み付けを出力できる
	* 不確実性を考慮できる（？）[Athiwaratkun et al., 2018]
		* point vector ではなく distribution に写像するので，uncertaintyが考慮できる…という理屈

## 提案手法
* Variational AutoEncoderの枠組みを採用
	* Encoder: Stacked Bi-LSTM
		* $x_{1:T} \rightarrow \{\alpha_t,\mu_t,\Sigma_t\}_{t=1}^{T}, p(z|x)=\sum_t{\alpha_t \phi(z;\mu_t,\Sigma_t)}$
	* Regularizer: approximated wasserstein distance
		* $d(p_{\theta}(z|x),q(z))$
		* $q(z) = \frac{1}{N_K}\sum_k{\phi(\mu_k,\sigma^2 I)}$ # prior distribution
	* Sampler: gumbel-softmax trick + reparametrization trick
		* $Z_i = \{z_{i,s}\}^{N_S}_{s=1}; z_{i,s} \sim p(z|x_i); N_S > 1$
		* mixture component choice を gumel-softmax で近似，gaussian sampler を reparametrization で代替
	* Decoder: Input-less LSTM[Bowman et al., 2016] + Attention
		* $p_\psi(x_{i,t}|Z_i) = softmax(MLP(h_t))$
		* $h_t, c_t = LSTM(v_t, h_{t-1})$
		* $v_t = Attn(q_t, Z_i)$ # {query, key&value} pair
		* $q_t = MLP([h_{t-1},c_{t-1}])$
	* Objective: reconstruction cost + regularization cost + smoothing cost (optional)
		* $l(\theta,\psi|x_i) = -E_{Z \sim p_\theta(z|x_i)}[lnp_{\psi}(x_i|Z)] + \gamma_d d(p_{\theta}(z|x_i),q(z)) + \gamma_\alpha KL(p_{\theta}(\alpha|x_i)||q(\alpha))$
* 文ペア $(x,y)$ の距離は，wasserstein distance を gaussian mixture に特化した近似計算(=approximated wasserstein distance)を採用
	1. distance matrix $K$ を解析的に計算[Takatsu 2011]
		* $k_{t,t'} = W_2^2(N(\mu_t^X,\Sigma_t^X), N(\mu_{t'}^Y,\Sigma_{t'}^Y)) = ||\mu_t^X-\mu_{t'}^Y||_2^2+tr(\Sigma_t^X+\Sigma_{t'}^Y-2((\Sigma_t^X)^{\frac{1}{2}} \Sigma_{t'}^Y (\Sigma_t^X)^{\frac{1}{2}})^{\frac{1}{2}} )$
		* covariance matrix が diagonal matrix の場合は，より簡潔な式になる
			* $W_2^2(N(\mu_t^X,diag(\sigma_t^X)), N(\mu_{t'}^Y, diag(\sigma_{t'}^Y))) = ||\mu_t^X-\mu_{t'}^Y||_2^2 + ||\sigma_t^X-\sigma_{t'}^Y||_2^2$
	2. 最適輸送問題を凸最適化に緩和したバージョンを，Sinkhorn algorithm[Cuturi 2013]を用いて計算
		* $d(x,y)=\sum_{t,t'}k_{t,t'}\pi^{*}_{t,t'}$
		* $\pi^{*}=min_{\pi \in \Pi(\alpha^X,\alpha^Y)}\{\sum_{t,t'}k_{t,t'}\pi_{t,t'}-\frac{1}{\lambda}H(\pi)\}$
	* Sinkhorn algorithmを用いると $\pi^{*}$ は線形演算の反復で求められる．従って微分可能性が担保される
* approximated wasserstein distance は，VAEの学習および，文間距離の推論の両方で用いる
	* VAEの学習： $d(p_{\theta}(z|x),q(z))$
	* 文間距離： $s(x,y)=d(p_{\theta}(z|x),p_{\theta}(z|y))$
