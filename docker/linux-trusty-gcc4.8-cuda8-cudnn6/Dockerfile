ARG BUILD_ID
FROM tensorcomprehensions/linux-trusty-gcc4.8:${BUILD_ID}

ARG BUILD
ADD ./common/install_cuda.sh install_cuda.sh
RUN bash ./install_cuda.sh

CMD ["bash"]
