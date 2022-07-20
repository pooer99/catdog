import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'
import ElementPlus from 'element-plus';
import 'element-plus/theme-chalk/index.css'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import axios from 'axios';
import VueAxios from 'vue-axios'

createApp(App).use(store).use(router).use(ElementPlus).use(VueAxios,axios).mount('#app')

const app = createApp(App)
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
    app.component(key, component)
}



