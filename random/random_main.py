import random
import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2

RAW_CAPTURE_MAX = (1 << 12) - 1
RAW_VIDEO_MAX = (1 << 10) - 1


def poisson_test():
    s = np.random.poisson(20, 1000000)
    count, bins, ignored = plt.hist(s, 15, density=True)
    plt.show()


# 参考google 论文 https://github.com/timothybrooks/unprocessing/blob/master/unprocess.py

# def random_noise_levels():
#   """Generates random noise levels from a log-log linear distribution."""
#   log_min_shot_noise = tf.log(0.0001)
#   log_max_shot_noise = tf.log(0.012)
#   log_shot_noise = tf.random_uniform((), log_min_shot_noise, log_max_shot_noise)
#   shot_noise = tf.exp(log_shot_noise)
#
#   line = lambda x: 2.18 * x + 1.20
#   log_read_noise = line(log_shot_noise) + tf.random_normal((), stddev=0.26)
#   read_noise = tf.exp(log_read_noise)
#   return shot_noise, read_noise
#
#
# def add_noise(image, shot_noise=0.01, read_noise=0.0005):
#   """Adds random shot (proportional to image) and read (independent) noise."""
#   variance = image * shot_noise + read_noise
#   noise = tf.random_normal(tf.shape(image), stddev=tf.sqrt(variance))
#   return image + noise
#
#
# def create_example(image):
#   """Creates training example of inputs and labels from `image`."""
#   image.shape.assert_is_compatible_with([None, None, 3])
#   image, metadata = unprocess.unprocess(image)
#   shot_noise, read_noise = unprocess.random_noise_levels()
#   noisy_img = unprocess.add_noise(image, shot_noise, read_noise)
#   # Approximation of variance is calculated using noisy image (rather than clean
#   # image), since that is what will be avaiable during evaluation.
#   variance = shot_noise * noisy_img + read_noise
#
#   inputs = {
#       'noisy_img': noisy_img,
#       'variance': variance,
#   }
#   inputs.update(metadata)
#   labels = image
#   return inputs, labels
#

def add_raw_noise():
    # 1. read original image
    filename = 'images/possion_test.png'
    img = cv2.imread(filename)
    # img = img.astype(float)
    # 2. calc sigmap using the calib parameter
    blc = 16
    img = correct_black_level(img, blc)
    # TODO: 减去黑电平，再计算sigmma map
    # TODO: 扣除ISO 50的poisson-gaussion分布，两个Poisson分布的和仍是Possion分布，均值和方差为2个分布之和；两个Gaussion分布的和仍是Gaussion分布，均值和方差为2个分布之和
    sigma_map = img * 0.1

    poisson_calib_para = [1, 2, 3]
    base_iso = 50
    target_iso = 3000
    # TODO: 待check，不需要ISO50的逻辑
    poisson_sigma_map_iso50 = cal_possion_sigma(img, base_iso, poisson_calib_para)
    # TODO: 这里img是否先减去ISO 50的噪声？
    poisson_sigma_map_target = cal_possion_sigma(img, target_iso, poisson_calib_para)
    get_poission_sigma_base_iso50(poisson_sigma_map_iso50, poisson_sigma_map_target)

    noisy_img = add_possion_noise(img, sigma_map)
    # TODO： 注意标准差要开方

    gaussian_calib_para = [1, 2, 3]
    target_gaussion_sigma = cal_gaussion_sigma(target_iso, gaussian_calib_para)
    iso50_gaussion_sigma = cal_gaussion_sigma(base_iso, gaussian_calib_para)
    gaussion_sigma = get_gaussion_sigma_base_iso50(iso50_gaussion_sigma, target_gaussion_sigma)
    noisy_img = add_gaussian_noise(noisy_img, gaussion_sigma)

    noisy_img = add_black_level(noisy_img, blc)

    cv2.imwrite("possion_noisy.png", noisy_img)


def cal_possion_sigma(image, target_iso, calib_para):
    return image * 0.1


def cal_gaussion_sigma(target_iso, calib_para):
    return 1


def get_poission_sigma_base_iso50(sigma_iso50, target_sigma):
    return target_sigma - sigma_iso50


def get_gaussion_sigma_base_iso50(sigma_iso50, target_sigma):
    return target_sigma - sigma_iso50


def correct_black_level(image, blc):
    return np.clip(image - blc, 0, RAW_CAPTURE_MAX)


def add_black_level(image, blc):
    return np.clip(image + blc, 0, RAW_CAPTURE_MAX)


def add_possion_noise(clean_image, sigma_map):
    # 3. generate possion noise
    noise_mask = np.random.poisson(sigma_map)
    # 4. add to the noisy free image to generate noisy image
    noisy_image = clean_image + noise_mask
    return noisy_image


def add_gaussian_noise(clean_image, sigma):
    noisy_image = np.random.normal(clean_image, sigma)
    round_img = np.round(noisy_image)
    return np.clip(round_img, 0, 255).astype(np.uint8)


def diff_possion_gaussion():
    from numpy import random
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.distplot(random.normal(loc=50, scale=7, size=1000), hist=False, label='normal')
    sns.distplot(random.poisson(lam=50, size=1000), hist=False, label='poisson')

    plt.show()


def diff_possion_binomial():
    from numpy import random
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.distplot(random.binomial(n=1000, p=0.01, size=1000), hist=False, label='binomial')
    sns.distplot(random.poisson(lam=10, size=1000), hist=False, label='poisson')

    plt.show()


def add_poisson_noise_test():
    # 1. read original image
    # filename = 'images/possion_test.png'
    # img = cv2.imread(filename)
    img = 128 * np.ones((128, 128), dtype=np.uint8).astype(np.uint8)
    cv2.imwrite("img_gray_clean.bmp", img)
    sigma_map = img * 0.1
    noisy_img = add_possion_noise(img, sigma_map)
    # TODO： 注意标准差要开方
    cv2.imwrite("img_gray_possion_noisy.png", noisy_img)

    # s = np.random.poisson(20, 1000000)
    s = noisy_img.reshape(128 * 128)
    count, bins, ignored = plt.hist(s, 15, density=True)
    plt.title("poisson noise distribution")
    plt.savefig("poisson_noise_distribution.png")
    plt.show()


def add_gaussian_noise_test():
    # 1. read original image
    # filename = 'images/possion_test.png'
    # img = cv2.imread(filename)
    img = 128 * np.ones((128, 128), dtype=np.uint8).astype(np.uint8)
    cv2.imwrite("img_gray_clean.bmp", img)
    sigma = 5
    noisy_img = add_gaussian_noise(img, sigma)
    cv2.imwrite("img_gray_gaussian_noisy.png", noisy_img)

    # s = np.random.poisson(20, 1000000)
    s = noisy_img.reshape(128 * 128)
    count, bins, ignored = plt.hist(s, 15, density=True)
    plt.title("gaussian noise distribution")
    plt.savefig("gaussian_noise_distribution.png")
    plt.show()


if __name__ == '__main__':
    # possion_test()
    # diff_possion_gaussion()
    # add_possion_noise()
    # add_poisson_noise_test()
    add_gaussian_noise_test()
