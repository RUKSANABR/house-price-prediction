import 'react-native-gesture-handler';
import * as React from 'react';
import {NavigationContainer} from '@react-navigation/native';
import {createNativeStackNavigator} from '@react-navigation/native-stack';
import Login from './src/pages/Login';
import SignUp from './src/pages/SignUp';
import Tabs from './src/pages/Tabs'
import Property from './src/pages/Property';
import ContracterPage from './src/pages/ContracterPage';
import Profile from './src/pages/Profile'
import Home from './src/components/user/Home';
import Notification from './src/pages/Notification';
const Stack = createNativeStackNavigator();

function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="Login" component={Login} />
        <Stack.Screen name="SignUp" component={SignUp} />
        <Stack.Screen name="Tabs" component={Tabs} options={{ headerShown: false }}/>
        <Stack.Screen name="Property" component={Property} options={{headerShown:false}}/>
        <Stack.Screen name="ContracterPage" component={ContracterPage} options={{headerShown:false}}/>
        <Stack.Screen name="Home" component={Home} options={{headerShown:false}}/>
        <Stack.Screen name="Profile" component={Profile} options={{headerShown:false}}/>
        <Stack.Screen name="Notification" component={Notification} options={{headerShown:false}}/>
      </Stack.Navigator>
    </NavigationContainer>
  );
}

export default App;
